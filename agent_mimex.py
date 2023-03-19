import numpy as np

import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim

from torch.distributions.categorical import Categorical

from bert import BERT
from model import CnnActorCriticNetwork, ICMModel

import wandb


class MIMExAgent(object):
    def __init__(
            self,
            input_size,
            output_size,
            num_env,
            num_step,
            gamma,
            lam=0.95,
            learning_rate=1e-4,
            ent_coef=0.01,
            clip_grad_norm=0.5,
            epoch=3,
            batch_size=128,
            ppo_eps=0.1,
            eta=1.,
            use_gae=True,
            use_cuda=False,
            use_noisy_net=False,
            expl_seq_len=5,
            n_mask=5,
            embed_dim=128,
            decoder_embed_dim=64,
            decoder_num_heads=2,
            decoder_depth=1,
            mask_ratio=0.7,
            bert_lr=1e-4):
        self.model = CnnActorCriticNetwork(input_size, output_size, use_noisy_net)
        self.num_env = num_env
        self.output_size = output_size
        self.input_size = input_size
        self.num_step = num_step
        self.gamma = gamma
        self.lam = lam
        self.epoch = epoch
        self.batch_size = batch_size
        self.use_gae = use_gae
        self.ent_coef = ent_coef
        self.eta = eta
        self.ppo_eps = ppo_eps
        self.clip_grad_norm = clip_grad_norm
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        self.n_mask = n_mask
        self.bert = BERT(
            seq_len=expl_seq_len,
            feature_dim=512,
            embed_dim=embed_dim,
            decoder_embed_dim=decoder_embed_dim,
            decoder_num_heads=decoder_num_heads,
            decoder_depth=decoder_depth,
            mask_ratio=mask_ratio).to(self.device)
        self.bert_opt = torch.optim.Adam(self.bert.parameters(), lr=bert_lr)

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.model = self.model.to(self.device)

    def get_action(self, state):
        state = torch.Tensor(state).to(self.device)
        state = state.float()
        feat, policy, value = self.model.forward_with_feat(state)
        action_prob = F.softmax(policy, dim=-1).data.cpu().numpy()

        action = self.random_choice_prob_index(action_prob)

        return feat, action, value.data.cpu().numpy().squeeze(), policy.detach()

    @staticmethod
    def random_choice_prob_index(p, axis=1):
        r = np.expand_dims(np.random.rand(p.shape[1 - axis]), axis=axis)
        return (p.cumsum(axis=axis) > r).argmax(axis=axis)

    def compute_intrinsic_reward(self, seq_feat):
        ### inference only
        # (num_worker, seq_len, 512)
        N = seq_feat.shape[0]
        bert_loss, _, _ = self.bert(seq_feat, keep_batch=True)
        if self.n_mask > 1:
            # mask multiple times and calculate average to reduce variance
            for _ in range(self.n_mask - 1):
                l, _, _ = self.bert(seq_feat, keep_batch=True)
                bert_loss += l
            bert_loss /= self.n_mask

        intrinsic_reward = self.eta * bert_loss.detach()
        return intrinsic_reward.data.cpu().numpy()

    def train_bert(self, seq_feat):
        ### training
        # (num_step, num_worker, seq_len, 512)
        M, N, T, D = seq_feat.shape
        bert_input = seq_feat.view(M * N, T, D)
        bert_loss, _, _ = self.bert(bert_input, keep_batch=True)
        if self.n_mask > 1:
            # mask multiple times and calculate average to reduce variance
            for _ in range(self.n_mask - 1):
                l, _, _ = self.bert(bert_input, keep_batch=True)
                bert_loss += l
            bert_loss /= self.n_mask

        # update BERT
        loss = bert_loss.mean()
        self.bert_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.bert_opt.step()

        return loss.item()

    def train_model(self, s_batch, next_s_batch, target_batch, y_batch, adv_batch, old_policy, seq_feat_batch):
        s_batch = torch.FloatTensor(s_batch).to(self.device)
        next_s_batch = torch.FloatTensor(next_s_batch).to(self.device)
        target_batch = torch.FloatTensor(target_batch).to(self.device)
        y_batch = torch.LongTensor(y_batch).to(self.device)
        adv_batch = torch.FloatTensor(adv_batch).to(self.device)

        sample_range = np.arange(len(s_batch))

        with torch.no_grad():
            policy_old_list = torch.stack(old_policy).permute(1, 0, 2).contiguous().view(-1, self.output_size).to(
                self.device)

            m_old = Categorical(F.softmax(policy_old_list, dim=-1))
            log_prob_old = m_old.log_prob(y_batch)
            # ------------------------------------------------------------

        total_bert_loss = 0.
        for i in range(self.epoch):
            np.random.shuffle(sample_range)
            for j in range(int(len(s_batch) / self.batch_size)):
                sample_idx = sample_range[self.batch_size * j:self.batch_size * (j + 1)]

                # --------------------------------------------------------------------------------
                # for MIMEx
                bert_loss = self.train_bert(seq_feat_batch)
                total_bert_loss += bert_loss
                # ---------------------------------------------------------------------------------

                policy, value = self.model(s_batch[sample_idx])
                m = Categorical(F.softmax(policy, dim=-1))
                log_prob = m.log_prob(y_batch[sample_idx])

                ratio = torch.exp(log_prob - log_prob_old[sample_idx])

                surr1 = ratio * adv_batch[sample_idx]
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.ppo_eps,
                    1.0 + self.ppo_eps) * adv_batch[sample_idx]

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(
                    value.sum(1), target_batch[sample_idx])

                entropy = m.entropy().mean()

                self.optimizer.zero_grad()
                loss = (actor_loss + 0.5 * critic_loss - 0.001 * entropy)
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

        return total_bert_loss / (self.epoch * int(len(s_batch) / self.batch_size))
