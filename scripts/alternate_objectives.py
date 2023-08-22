# Reward metric packages
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import single_meteor_score

# Reward function packages
import torch
import numpy as np
# from newsroom.analyze import Fragments
from scripts.fragments import Fragments


class TrainingRewardMetrics:
    def __init__(self):
        # ROUGE
        self.rouge = rouge_scorer.RougeScorer(rouge_types=['rougeL'], use_stemmer=False)

    @torch.no_grad()
    def train_metric_rouge(self, reference, prediction):
        rscore = self.rouge.score(reference, prediction)
        (P, R, F) = rscore['rougeL'][0], rscore['rougeL'][1], rscore['rougeL'][2]

        return P, R, F

    @staticmethod
    def train_metric_meteor(reference, prediction, decimal_places=4):
        mscore = round(single_meteor_score(reference, prediction), int(decimal_places))
        return mscore

    @staticmethod
    def train_metric_rouge_cov(input_doc, prediction, reference, beta=1):
        split_input = input_doc.split('|||||')
        split_input = [x for x in split_input if len(x) > 0]  # to only get text
        cov_scores = []
        cov_scores_ref = []
        for si in split_input:
            fragments_pred = Fragments(si, prediction)
            fragments_ref = Fragments(si, reference)
            cov_score_pred = fragments_pred.coverage()
            cov_score_ref = fragments_ref.coverage()
            cov_scores.append(cov_score_pred)
            cov_scores_ref.append(cov_score_ref)

        avg_pred = np.mean(cov_scores)
        avg_ref = np.mean(cov_scores_ref)
        std_pred = np.std(cov_scores)
        std_ref = np.std(cov_scores_ref)
        lhs_num = avg_pred / max(std_pred, 0.01)
        rhs_num = avg_ref / max(std_ref, 0.01)
        cov_reward_unnorm = (max(lhs_num, 0.01) - rhs_num) / max(lhs_num, 0.01)
        cov_reward = cov_reward_unnorm * (len(prediction.split()) / len(reference.split()))
        return cov_reward, cov_scores


def relax(logp, reference, q_func, log_temp):
    z, z_t, seq_tensor, y = gumbel_manipulation(logp)
    sparse = torch.zeros_like(z, dtype=torch.double).to(torch.device("cuda"))
    idx = torch.hstack([seq_tensor, reference[:, :-1].t()]).long().to(torch.device("cuda"))
    target_tensor = sparse.squeeze(0).index_put_(tuple(idx.t()), torch.tensor([1.0], dtype=torch.double).to(torch.device("cuda"))).unsqueeze(0).to(torch.device("cuda"))

    # need to convert model params to Double to prevent error
    q_func = q_func.double()
    if log_temp is not None:
        log_temp = torch.add(log_temp, 1e-8)
        temp = torch.exp(log_temp).unsqueeze(0).to(torch.device("cuda"))  # [B,T]
        # normalize
        sig_z = torch.softmax((z / temp), dim=-1).to(torch.device('cuda'))
        sig_zt = torch.softmax((z_t / temp), dim=-1).to(torch.device('cuda'))

        c_z = q_func(sig_z, target_tensor).mean(0)
        c_zt = q_func(sig_zt, target_tensor).mean(0)
    else:
        c_z = q_func(z, target_tensor).mean(0)
        c_zt = q_func(z_t, target_tensor).mean(0)
        log_temp = None

    # For use as a constant
    with torch.no_grad():
        c_zt_prime = c_zt.clone().detach().requires_grad_(False)

    return c_zt_prime, c_zt, c_z, y, log_temp


def sampling_overhead(logp):
    u = torch.rand(logp.shape).cuda()
    z = -torch.log(-torch.log(u + 1e-8) + 1e-8)
    y_soft = torch.softmax(torch.div((z+logp), 0.1), dim=-1)
    sample_y = torch.argmax(y_soft, dim=-1).detach().to(device="cuda")

    # create index tensor and gather values from lprobs using soft indices
    index_tensor = sample_y.T.unsqueeze(dim=0).T
    index_tensor = index_tensor.to(torch.int64)  # cast tensor to int64
    soft_lprobs = torch.gather(logp, 2, index_tensor).squeeze(-1)  # [B,T]

    return sample_y, soft_lprobs  # should all be GPU tensors


def gumbel_manipulation(logp):
    # 1. sample u and v from U(0,1)
    u = torch.FloatTensor(logp.shape).uniform_(1e-8, 1).to(torch.device('cuda'))
    v = torch.FloatTensor(logp.shape).uniform_(1e-8, 1).to(torch.device('cuda'))
    # 2. Manipulate to get z and y
    z = torch.add(-torch.log(-torch.log(u)), logp)  # phi in Eq 8 is the log probs, [B,T,V]
    y = torch.argmax(z, dim=-1)  # Gumbel-Max trick [B,T], indexes for max values
    # 3. Manipulate to get z_tilde
    vy = torch.gather(v, -1, torch.unsqueeze(y, -1))  # [B,T,1]
    z_tilde_y = -torch.log(-torch.log(vy))  # [B,T,1] tensor, location 0 on Gumbel distribution
    seq_tensor = torch.from_numpy(np.expand_dims(np.arange(0.0, list(v.shape)[1]), axis=0).transpose())  # [T,B]
    seq_tensor = seq_tensor.to(torch.device("cuda"))  # to GPU
    indices = torch.hstack([seq_tensor, y.t()]).long()
    probs = torch.exp(logp)
    probs = torch.add(probs, 1e-8)
    z_tilde_other = -torch.log(-torch.log(vy) - (torch.log(v) / probs))
    z_tilde = z_tilde_other.squeeze(0).index_put_(tuple(indices.t()), z_tilde_y.reshape(list(z_tilde_y.shape)[1]))
    z_tilde = z_tilde.reshape(logp.shape)
    return z, z_tilde, seq_tensor, y


class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_size=100):  # default
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, y):
        inp = torch.vstack([x, y])  # assumes target tensor is [T,V] sparse matrix
        hidden = self.fc1(inp)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = self.sigmoid(output)
        return output
