# <p align=center>`A Multi-Document Coverage Reward for RELAXed Multi-Document Summarization`</p>

This repository contains the implementation for our paper:
```
A Multi-Document Coverage Reward for RELAXed Multi-Document Summarization
Jacob Parnell, Inigo Jauregi Unanue, Massimo Piccardi
ACL 2022
```

## Code
`scripts/alternate_objectives.py`: Contains metrics used during training, and architecture for RELAX and gumbel sampling.

`scripts/xyz_dataset.py`: HuggingFace Datasets loader scripts (for fine-tuning).

`scripts/summarization.py`: Main file with model forward pass.

Links to the pre-trained Longformer models can be found in the `Longformer` repo.
Appendix A.2. in the paper provides links to the datasets used in these experiments.

To use the first 1000 samples for fine-tuning, we simply run:
```
head -1000 train.jsonl > train-1000.jsonl
```

### Run Code
```
# Run NLL pre-training
python3 scripts/summarization.py --save_dir='multinews_pretrain' --model_path='longformer-encdec-large-16384' --grad_ckpt --max_output_len 256 --dataset='multinews' --num_dataset_examples='all' --epochs 5 --custom_method='_nll'
```

```
# Run fine-tuning (e.g. ROUGE-L + Coverage with RELAX)
python3 scripts/summarization.py --lr 0.000003 --fine_tuning --save_dir="multinews_finetune" --model_path='longformer-encdec-large-16384' --from_pretrained='multinews_pretrain/test/{name_of_ckpt}.ckpt' --grad_ckpt --max_output_len 256 --dataset='multinews' --num_dataset_examples='1000' --epochs 1 --custom_method=rougecov1_relax
```

### Additional Notes
Due to the use of legacy `pytorch-lightning` code, and newer `torch` code, code must be changed in the backend to facilitate fine-tuning.

In `{virtual-env-name}/lib64/python3.6/site-packages/pytorch_lightning/core/saving.py`, to allow fine-tuning to run, change the following:
```
# Line 205 - remove **cls_kwargs
model = cls(*cls_args) #, **cls_kwargs)

# Line 207 - set strict=False when loading state_dict
model.load_state_dict(checkpoint['state_dict'], strict=False)
```

## Acknowledgements
This code is based off of the [Longformer](https://github.com/allenai/longformer) repo, and uses [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning)
We also utilise parts of the code from [RELAX](https://github.com/duvenaud/relax), and [Newsroom](https://github.com/lil-lab/newsroom) to implement the coverage reward.

## Citation
Please cite our work using the following:
```
```
