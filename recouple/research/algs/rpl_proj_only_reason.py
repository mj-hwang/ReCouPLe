import itertools
from typing import Any, Dict, Optional, Type

import torch
import torch.nn.functional as F

from research.networks.base import ActorCriticValueRewardPolicy

from .off_policy_algorithm import OffPolicyAlgorithm


def iql_loss(pred, target, expectile=0.5):
    err = target - pred
    weight = torch.abs(expectile - (err < 0).float())
    return weight * torch.square(err)

class RPLProjOnlyReason(OffPolicyAlgorithm):
    def __init__(
        self,
        *args,
        tau: float = 0.005,
        target_freq: int = 1,
        expectile: Optional[float] = None,
        beta: float = 1,
        clip_score: float = 100.0,
        reward_steps: Optional[int] = None,
        discount: float = 0.99,
        eval_tasks: list[int] = [],
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        assert isinstance(self.network, ActorCriticValueRewardPolicy)
        self.tau = tau
        self.target_freq = target_freq
        self.expectile = expectile
        self.beta = beta
        self.clip_score = clip_score
        self.reward_steps = reward_steps
        self.discount = discount
        self.eval_tasks = eval_tasks

        self.reward_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.reward_criterion_MSE = torch.nn.MSELoss(reduction="none")

        self.test_task_embedding = []

    def setup_network(self, network_class: Type[torch.nn.Module], network_kwargs: Dict) -> None:
        self.network = network_class(
            self.processor.observation_space, self.processor.action_space, **network_kwargs
        ).to(self.device)
        self.target_network = network_class(
            self.processor.observation_space, self.processor.action_space, **network_kwargs
        ).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        for param in self.target_network.parameters():
            param.requires_grad = False

    def setup_optimizers(self) -> None:
        # Default optimizer initialization
        network_keys = ("actor", "critic", "value", "reward")
        default_kwargs = {k: v for k, v in self.optim_kwargs.items() if k not in network_keys}
        assert all([isinstance(self.optim_kwargs.get(k, dict()), dict) for k in network_keys])

        # Update the encoder with the actor. This does better for weighted imitation policy objectives.
        actor_kwargs = default_kwargs.copy()
        actor_kwargs.update(self.optim_kwargs.get("actor", dict()))
        actor_params = itertools.chain(self.network.actor.parameters(), self.network.encoder.parameters())
        self.optim["actor"] = self.optim_class(actor_params, **actor_kwargs)

        critic_kwargs = default_kwargs.copy()
        critic_kwargs.update(self.optim_kwargs.get("critic", dict()))
        self.optim["critic"] = self.optim_class(self.network.critic.parameters(), **critic_kwargs)

        value_kwargs = default_kwargs.copy()
        value_kwargs.update(self.optim_kwargs.get("value", dict()))
        self.optim["value"] = self.optim_class(self.network.value.parameters(), **value_kwargs)

        reward_kwargs = default_kwargs.copy()
        reward_kwargs.update(self.optim_kwargs.get("reward", dict()))
        self.optim["reward"] = self.optim_class(self.network.reward.parameters(), **reward_kwargs)

    def train_step(self, batch: Dict, step: int, total_steps: int) -> Dict:
        assert isinstance(batch, dict) and "label" in batch, "Feedback batch must be used for efficient pref_iql"
        obs = torch.cat([batch["obs_1"], batch["obs_2"]], dim=0)  # (B, S+1)
        action = torch.cat([batch["action_1"], batch["action_2"]], dim=0)  # (B, S+1)
        # discount = torch.cat((batch["discount_1"], batch["discount_2"]), dim=0)  # (B, S+1)
        task_embeddings = torch.cat([batch["task_embeddings"], batch["task_embeddings"]], dim=0) 
        reason_embeddings = torch.cat([batch["reason_embeddings"], batch["reason_embeddings"]], dim=0)

        if step < self.reward_steps:
            self.network.reward.train()
            traj_embeddings = self.network.reward(obs, action) # (1, B, T, D)

            dot = torch.einsum('obtd,bd->obt', traj_embeddings, reason_embeddings) # (1, B, T)
            norm_squared = reason_embeddings.pow(2).sum(-1, keepdim=True).unsqueeze(0)  # (1, B, 1)
            reason_aligned_traj_embeddings = torch.einsum(
                'obt,bd->obtd', 
                dot / norm_squared,
                reason_embeddings
             ) # (1, B, T, D)
            reason_orthogonal_traj_embeddings = traj_embeddings - reason_aligned_traj_embeddings # (1, B, T, D)

            reason_aligned_reward = torch.einsum(
                'obtd,bd->obt', 
                reason_aligned_traj_embeddings, 
                task_embeddings
            )  # (1, B, T)
            reason_orthogonal_reason = torch.einsum(
                'obtd,bd->obt', 
                reason_orthogonal_traj_embeddings, 
                task_embeddings
            )  # (1, B, T)

            rar1, rar2 = torch.chunk(reason_aligned_reward.sum(dim=-1), 2, dim=1)
            ror1, ror2 = torch.chunk(reason_orthogonal_reason.sum(dim=-1), 2, dim=1)

            # reward
            reward = reason_aligned_reward + reason_orthogonal_reason
            r1 = rar1 + ror1
            r2 = rar2 + ror2

            rar_logits = rar2 - rar1
            labels = batch["label"].float().unsqueeze(0).expand_as(rar_logits)
            assert labels.shape == rar_logits.shape
            rar_BT_loss = self.reward_criterion(rar_logits, labels).mean()
            
            with torch.no_grad():
                reward_accuracy = ((r2 > r1) == torch.round(labels)).float().mean()
                reason_aligned_reward_accuracy = ((rar2 > rar1) == torch.round(labels)).float().mean()

            loss = rar_BT_loss

            self.optim["reward"].zero_grad(set_to_none=True)
            loss.backward()
            self.optim["reward"].step()

            reward = reward.detach().mean(dim=0)
        else:
            with torch.no_grad():
                traj_embeddings = self.network.reward(obs, action)
                reward = torch.einsum('obtd,bd->obt', traj_embeddings, task_embeddings).mean(dim=0)

        metrics = dict()
        if step < self.reward_steps:
            metrics["rar_BT_loss"] = rar_BT_loss.item()
            metrics["reward_accuracy"] = reward_accuracy.item()
            metrics["reason_aligned_reward_accuracy"] = reason_aligned_reward_accuracy.item()

        # Update the networks. These are done in a stack to support different grad options for the encoder.
        if step % self.target_freq == 0:
            with torch.no_grad():
                # Only run on the critic and encoder, those are the only weights we update.
                for param, target_param in zip(
                    self.network.critic.parameters(), self.target_network.critic.parameters()
                ):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return metrics

    def validation_step(self, batch: Dict) -> Dict:
        # Compute the loss
        print("validation called")
        metrics = {}
        with torch.no_grad():
            obs_1 = batch["obs_1"]
            obs_2 = batch["obs_2"]

            action_1 = batch["action_1"]
            action_2 = batch["action_2"]

            task_labels = batch["task_labels"]
            task_embeddings = batch["task_embeddings"]

            for i in self.eval_tasks:
                traj_embeddings_t1 = self.network.reward(obs_1[task_labels==i], action_1[task_labels==i])
                reward_t1 = torch.einsum('obtd,bd->obt', traj_embeddings_t1, task_embeddings[task_labels==i])

                traj_embeddings_t2 = self.network.reward(obs_2[task_labels==i], action_2[task_labels==i])
                reward_t2 = torch.einsum('obtd,bd->obt', traj_embeddings_t2, task_embeddings[task_labels==i])
                
                r_t1 = reward_t1.sum(dim=-1)
                r_t2 = reward_t2.sum(dim=-1)
                labels = batch["label"][task_labels==i].float().unsqueeze(0).expand_as(r_t1)
                task_accuracy = ((r_t2 > r_t1) == torch.round(labels)).float().mean()
                metrics[f"task_{i}_accuracy"] = task_accuracy.item()

        return metrics

    def _get_train_action(self, obs: Any, step: int, total_steps: int):
        batch = dict(obs=obs)
        with torch.no_grad():
            action = self.predict(batch, is_batched=False, sample=True)
        return action
