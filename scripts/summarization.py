import os
import argparse
import random
import numpy as np
import logging

import torch
from torch.utils.data import DataLoader, Dataset
import nlp
from datasets import load_dataset
from rouge_score import rouge_scorer

import pytorch_lightning as pl
from pytorch_lightning.loggers import TestTubeLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.overrides.data_parallel import LightningDistributedDataParallel

from longformer.longformer_encoder_decoder import LongformerEncoderDecoderForConditionalGeneration, \
    LongformerEncoderDecoderConfig
from longformer.sliding_chunks import pad_to_window_size

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
from transformers.optimization import get_linear_schedule_with_warmup, Adafactor

from scripts.alternate_objectives import TrainingRewardMetrics, relax, sampling_overhead, Feedforward

import warnings
warnings.filterwarnings("ignore")

# Initialise for custom training
trm = TrainingRewardMetrics()


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100):
    """From fairseq"""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
        count = (~pad_mask).sum()
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
        count = nll_loss.numel()

    nll_loss = nll_loss.sum() / count
    smooth_loss = smooth_loss.sum() / count
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


def log_writer(out_file, output_dict):
    """To log outputs"""
    if not os.path.exists(out_file):
        with open(out_file, "w") as f:
            for k, v in output_dict.items():
                f.write("%s %s\n" % (k, v))
            f.write("-----\n")

    else:  # append
        with open(out_file, "a") as f:
            for k, v in output_dict.items():
                f.write("%s %s\n" % (k, v))
            f.write("-----\n")


class SummarizationDataset(Dataset):
    def __init__(self, ds_name, hf_dataset, tokenizer, max_input_len, max_output_len):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.ds_name = ds_name

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        entry = self.hf_dataset[idx]

        if self.ds_name == "multinews":
            # nll pre-training
            # input_ids = self.tokenizer.encode(entry['document'], truncation=True, max_length=self.max_input_len)
            # output_ids = self.tokenizer.encode(entry['summary'], truncation=True, max_length=self.max_output_len)
            # fine-tuning
            input_ids = self.tokenizer.encode(entry['src_str'], truncation=True, max_length=self.max_input_len)
            output_ids = self.tokenizer.encode(entry['tgt_str'], truncation=True, max_length=self.max_output_len)

        elif self.ds_name == "wcep":
            # WCEP
            input_ids = self.tokenizer.encode(entry['src_str'], truncation=True, max_length=self.max_input_len)
            output_ids = self.tokenizer.encode(entry['tgt_str'], truncation=True, max_length=self.max_output_len)

        else:
            raise ValueError("Dataset name does not match the examples here")

        if self.tokenizer.bos_token_id is None:  # pegasus
            output_ids = [self.tokenizer.pad_token_id] + output_ids
        return torch.tensor(input_ids), torch.tensor(output_ids)

    @staticmethod
    def collate_fn(batch):
        # A hack to know if this is bart or pegasus. DDP doesn't like global variables nor class-level memebr variables
        if batch[0][0][-1].item() == 2:
            pad_token_id = 1  # AutoTokenizer.from_pretrained('facebook/bart-base').pad_token_id
        elif batch[0][0][-1].item() == 1:
            pad_token_id = 0  # AutoTokenizer.from_pretrained('google/pegasus-large').pad_token_id
        else:
            assert False

        input_ids, output_ids = list(zip(*batch))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        output_ids = torch.nn.utils.rnn.pad_sequence(output_ids, batch_first=True, padding_value=pad_token_id)
        return input_ids, output_ids


class Summarizer(pl.LightningModule):

    def __init__(self, params):
        super().__init__()
        self.args = params
        # self.hparams = params
        self.tokenizer = AutoTokenizer.from_pretrained(self.args['tokenizer'], use_fast=True)

        if 'long' in self.args['model_path']:
            config = LongformerEncoderDecoderConfig.from_pretrained(self.args['model_path'])
            config.attention_dropout = self.args['attention_dropout']
            config.gradient_checkpointing = self.args['grad_ckpt']
            config.attention_mode = self.args['attention_mode']
            config.attention_window = [self.args['attention_window']] * config.encoder_layers
            self.model = LongformerEncoderDecoderForConditionalGeneration.from_pretrained(
                self.args['model_path'], config=config)
        else:
            config = AutoConfig.from_pretrained(self.args.model_path)
            config.attention_dropout = self.args.attention_dropout
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.args.model_path, config=config)
        self.train_dataloader_object = self.val_dataloader_object = self.test_dataloader_object = None

        if "relax" in self.args['custom_method']:
            self.qfunc = Feedforward(len(self.tokenizer), 256)  # 64)
            self.log_temp = torch.from_numpy(np.array([0.5] * len(self.tokenizer), dtype=np.float32))
            self.log_temp.requires_grad_(True)
            self.log_temp_exists = True

            self.model.state_dict()["qfunc.fc1.weight"] = list(self.qfunc.parameters())[0]
            self.model.state_dict()["qfunc.fc1.bias"] = list(self.qfunc.parameters())[1]
            self.model.state_dict()["qfunc.fc2.weight"] = list(self.qfunc.parameters())[2]
            self.model.state_dict()["qfunc.fc2.bias"] = list(self.qfunc.parameters())[3]

    def _prepare_input(self, input_ids):
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
        attention_mask[input_ids == self.tokenizer.pad_token_id] = 0
        if isinstance(self.model, LongformerEncoderDecoderForConditionalGeneration):
            attention_mask[:, 0] = 2  # global attention on one token for all model params to be used, which is important for gradient checkpointing to work
            if self.args['attention_mode'] == 'sliding_chunks':
                half_padding_mod = self.model.config.attention_window[0]
            elif self.args['attention_mode'] == 'sliding_chunks_no_overlap':
                half_padding_mod = self.model.config.attention_window[0] / 2
            else:
                raise NotImplementedError
            input_ids, attention_mask = pad_to_window_size(  # ideally, should be moved inside the LongformerModel
                input_ids, attention_mask, half_padding_mod, self.tokenizer.pad_token_id)
        return input_ids, attention_mask

    def forward(self, input_ids, output_ids):
        input_ids, attention_mask = self._prepare_input(input_ids)
        decoder_input_ids = output_ids[:, :-1]
        decoder_attention_mask = (decoder_input_ids != self.tokenizer.pad_token_id)
        labels = output_ids[:, 1:].clone()
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            use_cache=False, )
        lm_logits = outputs[0]

        lprobs = torch.nn.functional.log_softmax(lm_logits, dim=-1)

        # Determine custom method
        metric, method = self.args['custom_method'].split('_')
        if "nll" not in method:
            logging.info(f"Custom training with: {metric} and {method}")
            # Sampling
            sample_ids, soft_lprobs = sampling_overhead(lprobs)
            pad_mask = sample_ids.eq(self.tokenizer.pad_token_id)
            soft_lprobs.masked_fill_(pad_mask, 0.0)

            # Decode
            # These are in batches so we return first element as this is a batch_size=1 run
            reference = self.tokenizer.batch_decode(labels.tolist(), skip_special_tokens=True)[0]
            sampled_prediction = self.tokenizer.batch_decode(sample_ids.tolist(), skip_special_tokens=True)[0]

            # Reward metric
            with torch.no_grad():
                if metric == "rouge":
                    soft_p, soft_r, soft_f = trm.train_metric_rouge(reference, sampled_prediction)
                elif "rougecov" in metric:
                    input_doc = self.tokenizer.batch_decode(input_ids.tolist(), skip_special_tokens=True)[0]
                    rouge = rouge_scorer.RougeScorer(rouge_types=['rougeL'], use_stemmer=False)
                    rscore = rouge.score(reference, sampled_prediction)
                    rl = rscore['rougeL'][2]  # ROUGE-L F1
                    cov_reward, cov_scores = trm.train_metric_rouge_cov(input_doc, sampled_prediction, reference)
                    beta = 1.0
                    soft_f = rl + beta*cov_reward
                else:
                    raise ValueError("No reward metric specified")

            # Reward function
            if method == "reinforce":
                loss = -soft_f*soft_lprobs.sum()

            elif method == "relax":
                c_zt_prime, c_zt, c_z, gumbel_y, log_temp = relax(lprobs, output_ids, self.qfunc, self.log_temp)
                gumbel_prediction = self.tokenizer.batch_decode(gumbel_y.tolist(), skip_special_tokens=True)[0]
                # Find the reward using the gumbel y
                with torch.no_grad():
                    if metric == "rouge":
                        soft_p, soft_r, soft_f = trm.train_metric_rouge(reference, gumbel_prediction)
                    elif "rougecov" in metric:
                        input_doc = self.tokenizer.batch_decode(input_ids.tolist(), skip_special_tokens=True)[0]
                        rouge = rouge_scorer.RougeScorer(rouge_types=['rougeL'], use_stemmer=False)
                        rscore = rouge.score(reference, gumbel_prediction)
                        rl = rscore['rougeL'][2]  # ROUGE-L F1
                        cov_reward, cov_scores = trm.train_metric_rouge_cov(input_doc, gumbel_prediction, reference)
                        beta = 1.0
                        soft_f = rl + beta * cov_reward

                lprobs_gumbel_y = torch.gather(lprobs, -1, gumbel_y.t().unsqueeze(0)).squeeze(-1)
                lprobs_gumbel_y = torch.add(lprobs_gumbel_y, 1e-8)
                relax_loss = torch.mul((soft_f - c_zt_prime.unsqueeze(0).squeeze(2)), lprobs_gumbel_y) - c_zt.unsqueeze(0).squeeze(2) + c_z.unsqueeze(0).squeeze(2)
                relax_loss = relax_loss.mean(1)  # average across T (loss per token)

            else:
                loss = 0
                raise TypeError

        # Default NLL loss
        elif self.args['label_smoothing'] == 0:
            # Same behavior as modeling_bart.py, besides ignoring pad_token_id
            ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
            assert lm_logits.shape[-1] == self.model.config.vocab_size
            loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), labels.view(-1))

        else:
            lprobs = torch.nn.functional.log_softmax(lm_logits, dim=-1)
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs, labels, self.args['label_smoothing'], ignore_index=self.tokenizer.pad_token_id
            )

        if "relax" in method:  # RELAX
            return [relax_loss, lm_logits]
        else:  # otherwise
            return [loss]

    def training_step(self, batch, batch_nb, optimizer_idx):
        for p in self.model.parameters():
            p.requires_grad = True
        if "relax" in self.args["custom_method"]:
            for p in list(self.qfunc.parameters()):
                p.requires_grad = True
            self.log_temp.requires_grad_(True)
        torch.set_grad_enabled(True)

        output = self.forward(*batch)
        loss = output[0]
        print(f"{self.args['custom_method'].split('_')[1]}_loss: {loss}")
        # Tail end of RELAX implementation -> in order of optimizers
        if "relax" in self.args["custom_method"] and optimizer_idx == 1:
            lm_logits = output[1]
            grad_relax_loss = torch.autograd.grad(loss, lm_logits, grad_outputs=torch.ones_like(loss),
                                                  create_graph=True, retain_graph=True)[0]
            loss = torch.mean(torch.square(grad_relax_loss))
            print(f"var_loss: {loss}")

        lr = loss.new_zeros(1) + self.trainer.optimizers[0].param_groups[0]['lr']
        tensorboard_logs = {'train_loss': loss, 'lr': lr,
                            'input_size': batch[0].numel(),
                            'output_size': batch[1].numel(),
                            'mem': torch.cuda.memory_allocated(loss.device) / 1024 ** 3 if torch.cuda.is_available() else 0}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb, test=False):
        for p in self.model.parameters():
            p.requires_grad = False
        if "relax" in self.args["custom_method"]:
            for p in list(self.qfunc.parameters()):
                p.requires_grad = False
            self.log_temp.requires_grad_(False)

        outputs = self.forward(*batch)
        vloss = outputs[0]
        input_ids, output_ids = batch
        input_ids, attention_mask = self._prepare_input(input_ids)
        generated_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                            use_cache=True, max_length=self.args['max_output_len'],
                                            num_beams=1)
        # Metrics
        generated_str = self.tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=True)
        gold_str = self.tokenizer.batch_decode(output_ids.tolist(), skip_special_tokens=True)
        scorer = rouge_scorer.RougeScorer(rouge_types=['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=False)
        rouge1 = rouge2 = rougel = rougelsum = 0.0
        meteor = 0.0
        for ref, pred in zip(gold_str, generated_str):
            score = scorer.score(ref, pred)
            meteor_score = trm.train_metric_meteor(ref, pred)
            rouge1 += score['rouge1'].fmeasure
            rouge2 += score['rouge2'].fmeasure
            rougel += score['rougeL'].fmeasure
            rougelsum += score['rougeLsum'].fmeasure
            meteor += meteor_score
        rouge1 /= len(generated_str)
        rouge2 /= len(generated_str)
        rougel /= len(generated_str)
        rougelsum /= len(generated_str)
        meteor /= len(generated_str)

        input_str = self.tokenizer.batch_decode(input_ids.tolist(), skip_special_tokens=True)
        out_prefix = 'test' if test else 'val'  # separate test and val
        out_file = os.path.join(self.args['save_dir'], f"text_metrics-{out_prefix}.txt")

        text_dict = {
            "INPUT: ": input_str,
            "TARGET: ": gold_str,
            "PRED: ": generated_str,
            "SCORE: ": f"{rouge1} / {rouge2} / {rougel} / {rougelsum} / {meteor}"
        }
        log_writer(out_file, text_dict)

        return {'vloss': vloss,
                'rouge1': vloss.new_zeros(1) + rouge1,
                'rouge2': vloss.new_zeros(1) + rouge2,
                'rougeL': vloss.new_zeros(1) + rougel,
                'rougeLsum': vloss.new_zeros(1) + rougelsum,
                'meteor_score': vloss.new_zeros(1) + meteor}

    def validation_epoch_end(self, outputs, test=False):
        for p in self.model.parameters():
            p.requires_grad = True
        if "relax" in self.args["custom_method"]:
            for p in list(self.qfunc.parameters()):
                p.requires_grad = True
            self.log_temp.requires_grad_(True)

        names = ['vloss', 'rouge1', 'rouge2', 'rougeL', 'rougeLsum', 'meteor_score']
        metrics = []
        for name in names:
            metric = torch.stack([x[name] for x in outputs]).mean()
            if self.trainer.use_ddp:
                torch.distributed.all_reduce(metric, op=torch.distributed.ReduceOp.SUM)
                metric /= self.trainer.world_size
            metrics.append(metric)
        logs = dict(zip(*[names, metrics]))
        print(logs)

        out_prefix = 'test' if test else 'val'  # separate test and val
        out_file = os.path.join(self.args['save_dir'], f"text_metrics-epoch-end-{out_prefix}.txt")
        log_writer(out_file, logs)

        return {'avg_val_loss': logs['vloss'], 'log': logs, 'progress_bar': logs}

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb, test=True)

    def test_epoch_end(self, outputs):
        result = self.validation_epoch_end(outputs, test=True)
        print(result)

    def configure_optimizers(self):
        if self.args['adafactor']:
            optimizer = Adafactor(self.model.parameters(), lr=self.args['lr'], scale_parameter=False, relative_step=False)
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args['lr'])
        if self.args['debug']:
            return optimizer  # const LR
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        num_steps = self.args['dataset_size'] * self.args['epochs'] / num_gpus / self.args['grad_accum'] / self.args['batch_size']
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args['warmup'], num_training_steps=num_steps
        )
        # RELAX
        if "relax" in self.args['custom_method']:
            if self.log_temp_exists:
                additional_params = list(self.qfunc.parameters())
                additional_params.append(self.log_temp)
                tune_optim = torch.optim.Adam(additional_params, lr=0.01)
            else:
                tune_optim = torch.optim.Adam(list(self.qfunc.parameters()), lr=0.01)
            return [optimizer, tune_optim], [{"scheduler": scheduler, "interval": "step"}]
        else:
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def _get_dataloader(self, current_dataloader, split_name, is_train):
        if current_dataloader is not None:
            return current_dataloader
        dataset = SummarizationDataset(ds_name=self.name_dataset, hf_dataset=self.hf_datasets[split_name],
                                       tokenizer=self.tokenizer, max_input_len=self.args['max_input_len'],
                                       max_output_len=self.args['max_output_len'])
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=is_train) if self.trainer.use_ddp else None
        return DataLoader(dataset, batch_size=self.args['batch_size'], shuffle=(sampler is None),
                          num_workers=self.args['num_workers'], sampler=sampler,
                          collate_fn=SummarizationDataset.collate_fn)

    @pl.data_loader
    def train_dataloader(self):
        self.train_dataloader_object = self._get_dataloader(self.train_dataloader_object, 'train', is_train=True)
        return self.train_dataloader_object

    @pl.data_loader
    def val_dataloader(self):
        self.val_dataloader_object = self._get_dataloader(self.val_dataloader_object, 'validation', is_train=False)
        return self.val_dataloader_object

    @pl.data_loader
    def test_dataloader(self):
        self.test_dataloader_object = self._get_dataloader(self.test_dataloader_object, 'test', is_train=False)
        return self.test_dataloader_object

    def configure_ddp(self, model, device_ids):
        model = LightningDistributedDataParallel(
            model,
            device_ids=device_ids,
            find_unused_parameters=False
        )
        return model

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        parser.add_argument("--save_dir", type=str, default='summarization')
        parser.add_argument("--save_prefix", type=str, default='test')
        parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
        parser.add_argument("--grad_accum", type=int, default=1, help="number of gradient accumulation steps")
        parser.add_argument("--gpus", type=int, default=-1,
                            help="Number of gpus. 0 for CPU")
        parser.add_argument("--warmup", type=int, default=1000, help="Number of warmup steps")
        parser.add_argument("--lr", type=float, default=0.00003, help="Maximum learning rate")
        parser.add_argument("--val_every", type=float, default=1.0, help="Number of training steps between validations")
        parser.add_argument("--val_percent_check", default=1.00, type=float, help='Percent of validation data used')
        parser.add_argument("--num_workers", type=int, default=0, help="Number of data loader workers")
        parser.add_argument("--seed", type=int, default=1234, help="Seed")
        parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
        parser.add_argument("--disable_checkpointing", action='store_true', help="No logging or checkpointing")
        parser.add_argument("--max_output_len", type=int, default=64,
                            help="maximum num of wordpieces/summary. Used for training and testing")
        parser.add_argument("--max_input_len", type=int, default=16384,
                            help="maximum num of wordpieces/summary. Used for training and testing")
        parser.add_argument("--test", action='store_true', help="Test only, no training")
        parser.add_argument("--model_path", type=str, default='facebook/bart-base',
                            help="Path to the checkpoint directory or model name")
        parser.add_argument("--tokenizer", type=str, default='facebook/bart-base')
        parser.add_argument("--no_progress_bar", action='store_true', help="no progress bar. Good for printing")
        parser.add_argument("--fp32", action='store_true', help="default is fp16. Use --fp32 to switch to fp32")
        parser.add_argument("--debug", action='store_true', help="debug run")
        parser.add_argument("--resume_ckpt", type=str, help="Path of a checkpoint to resume from")
        parser.add_argument("--from_pretrained", type=str, default=None,
                            help="Path to a checkpoint to load model weights but not training state")
        parser.add_argument('--grad_ckpt', action='store_true', help='Enable gradient checkpointing to save memory')
        parser.add_argument("--attention_dropout", type=float, default=0.1, help="attention dropout")
        parser.add_argument("--attention_mode", type=str, default='sliding_chunks', help="Longformer attention mode")
        parser.add_argument("--attention_window", type=int, default=512, help="Attention window")
        parser.add_argument("--label_smoothing", type=float, default=0.0, required=False)
        parser.add_argument("--adafactor", action='store_true', help="Use adafactor optimizer")
        parser.add_argument("--custom_method", type=str, default=None, help="Custom training implementation")
        parser.add_argument("--dataset", type=str, default=None, help="Custom dataset choice")
        parser.add_argument("--num_dataset_examples", type=str, default=None, help="Custom num dataset examples")
        parser.add_argument("--fine_tuning", action="store_true", help="Custom fine-tuning flag")

        return parser


def main(args):
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Current GPU device: {torch.cuda.current_device()}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.from_pretrained is not None:
        model = Summarizer.load_from_checkpoint(args.from_pretrained, vars(args))
    else:
        model = Summarizer(vars(args))

    model.name_dataset = args.dataset

    # Load data
    if args.dataset == "multinews":
        if args.num_dataset_examples == 'all':
            model.hf_datasets = nlp.load_dataset('multi_news', cache_dir='cache_dir')
            args.dataset_size = 50594  # train + val
        elif args.num_dataset_examples != 'all' and args.fine_tuning:  # fine-tuning
            model.hf_datasets = load_dataset(f"scripts/mn_dataset_{int(args.num_dataset_examples)}ex.py")
            args.dataset_size = int(args.num_dataset_examples) + 5622
    elif args.dataset == "wcep":
        if args.num_dataset_examples == 'all':
            model.hf_datasets = load_dataset("scripts/wcep_dataset.py", cache_dir='cache_dir')
            args.dataset_size = 9178  # train + val
        elif args.num_dataset_examples != 'all' and args.fine_tuning:  # fine-tuning
            model.hf_datasets = load_dataset(f"scripts/wcep_dataset_{int(args.num_dataset_examples)}ex.py",
                                             cache_dir='cache_dir')
            args.dataset_size = int(args.num_dataset_examples) + 1020
    else:
        raise ValueError("Need to specify a dataset")
    logging.info(model.hf_datasets)

    logger = TestTubeLogger(
        save_dir=args.save_dir,
        name=args.save_prefix,
        version=0
    )

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(args.save_dir, args.save_prefix, "checkpoints"),
        save_top_k=5,
        verbose=True,
        monitor='avg_val_loss',
        mode='min',
        period=-1,
        prefix=''
    )

    print(args)

    trainer = pl.Trainer(gpus=args.gpus, distributed_backend='ddp' if torch.cuda.is_available() else None,
                         track_grad_norm=-1,
                         max_epochs=args.epochs if not args.debug else 100,
                         max_steps=None if not args.debug else 1,
                         replace_sampler_ddp=False,
                         accumulate_grad_batches=args.grad_accum,
                         val_check_interval=args.val_every if not args.debug else 1,
                         num_sanity_val_steps=2 if not args.debug else 0,
                         check_val_every_n_epoch=1 if not args.debug else 1,
                         val_percent_check=args.val_percent_check,
                         test_percent_check=args.val_percent_check,
                         logger=logger,
                         checkpoint_callback=checkpoint_callback if not args.disable_checkpointing else False,
                         progress_bar_refresh_rate=1, show_progress_bar=not args.no_progress_bar,
                         use_amp=False,
                         amp_level='O2',
                         resume_from_checkpoint=args.resume_ckpt,
                         )
    if not args.test:
        trainer.fit(model)
    trainer.test(model)


if __name__ == "__main__":
    main_arg_parser = argparse.ArgumentParser(description="summarization")
    parser = Summarizer.add_model_specific_args(main_arg_parser, os.getcwd())
    args = parser.parse_args()
    main(args)
