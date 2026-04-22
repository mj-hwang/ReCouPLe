import itertools
from typing import Any, Dict, Optional, Type

import torch

from research.networks.base import ActorCriticValueRewardPolicy

from .off_policy_algorithm import OffPolicyAlgorithm


def iql_loss(pred, target, expectile=0.5):
    err = target - pred
    weight = torch.abs(expectile - (err < 0).float())
    return weight * torch.square(err)


# class MTPIQL(OffPolicyAlgorithm):
#     def __init__(
#         self,
#         *args,
#         tau: float = 0.005,
#         target_freq: int = 1,
#         expectile: Optional[float] = None,
#         beta: float = 1,
#         clip_score: float = 100.0,
#         reward_steps: Optional[int] = None,
#         discount: float = 0.99,
#         eval_tasks: list[int] = [],
#         **kwargs,
#     ) -> None:
#         super().__init__(*args, **kwargs)
#         assert isinstance(self.network, ActorCriticValueRewardPolicy)
#         self.tau = tau
#         self.target_freq = target_freq
#         self.expectile = expectile
#         self.beta = beta
#         self.clip_score = clip_score
#         self.reward_steps = reward_steps
#         self.discount = discount
#         self.reward_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
#         self.eval_tasks = eval_tasks

#     def setup_network(self, network_class: Type[torch.nn.Module], network_kwargs: Dict) -> None:
#         self.network = network_class(
#             self.processor.observation_space, self.processor.action_space, **network_kwargs
#         ).to(self.device)
#         self.target_network = network_class(
#             self.processor.observation_space, self.processor.action_space, **network_kwargs
#         ).to(self.device)
#         self.target_network.load_state_dict(self.network.state_dict())
#         for param in self.target_network.parameters():
#             param.requires_grad = False

#     def setup_optimizers(self) -> None:
#         # Default optimizer initialization
#         # network_keys = ("actor", "critic", "value", "trajectory_encoder", "task_encoder")
#         network_keys = ("actor", "critic", "value", "reward")
#         default_kwargs = {k: v for k, v in self.optim_kwargs.items() if k not in network_keys}
#         assert all([isinstance(self.optim_kwargs.get(k, dict()), dict) for k in network_keys])

#         # Update the encoder with the actor. This does better for weighted imitation policy objectives.
#         actor_kwargs = default_kwargs.copy()
#         actor_kwargs.update(self.optim_kwargs.get("actor", dict()))
#         actor_params = itertools.chain(self.network.actor.parameters(), self.network.encoder.parameters())
#         self.optim["actor"] = self.optim_class(actor_params, **actor_kwargs)

#         critic_kwargs = default_kwargs.copy()
#         critic_kwargs.update(self.optim_kwargs.get("critic", dict()))
#         self.optim["critic"] = self.optim_class(self.network.critic.parameters(), **critic_kwargs)

#         value_kwargs = default_kwargs.copy()
#         value_kwargs.update(self.optim_kwargs.get("value", dict()))
#         self.optim["value"] = self.optim_class(self.network.value.parameters(), **value_kwargs)

#         reward_kwargs = default_kwargs.copy()
#         reward_kwargs.update(self.optim_kwargs.get("reward", dict()))
#         self.optim["reward"] = self.optim_class(self.network.reward.parameters(), **reward_kwargs)

#         # trajectory_encoder_kwargs = default_kwargs.copy()
#         # trajectory_encoder_kwargs.update(self.optim_kwargs.get("trajectory_encoder", dict()))
#         # self.optim["trajectory_encoder"] = self.optim_class(self.network.trajectory_encoder.parameters(), **trajectory_encoder_kwargs)

#         # task_encoder_kwargs = default_kwargs.copy()
#         # task_encoder_kwargs.update(self.optim_kwargs.get("task_encoder", dict()))
#         # self.optim["task_encoder"] = self.optim_class(self.network.task_encoder.parameters(), **task_encoder_kwargs)

#     def train_step(self, batch: Dict, step: int, total_steps: int) -> Dict:
#         assert isinstance(batch, dict) and "label" in batch, "Feedback batch must be used for efficient pref_iql"
#         obs = torch.cat([batch["obs_1"], batch["obs_2"]], dim=0)  # (B, S+1)
#         action = torch.cat([batch["action_1"], batch["action_2"]], dim=0)  # (B, S+1)
#         obs_1 = batch["obs_1"]
#         obs_2 = batch["obs_2"]

#         action_1 = batch["action_1"]
#         action_2 = batch["action_2"]

#         task_tokens = batch["task_tokens"]
#         task_masks = batch["task_masks"]

#         # breakpoint()

#         if "discount_1" in batch:
#             discount = torch.cat((batch["discount_1"], batch["discount_2"]), dim=0)  # (B, S+1)
#         else:
#             discount = self.discount

#         if step < self.reward_steps:
#             print(step)
#             self.network.reward.train()

#             # reward = self.network.reward(obs, action)
#             # traj_feature = self.network.trajectory_encoder(obs, action)
#             # task_feature = self.network.task_encoder(task_tokens, task_masks)
#             # reward = traj_feature * task_feature

#             # reward = self.network.reward(obs, action, task_tokens, task_masks)
#             # r1, r2 = torch.chunk(reward.sum(dim=-1), 2, dim=1)  # Should return two (E, B)

#             reward_1 = self.network.reward(obs_1, action_1, task_tokens, task_masks)
#             reward_2 = self.network.reward(obs_2, action_2, task_tokens, task_masks)
#             reward = torch.cat([reward_1, reward_2], dim=1)
            
#             r1 = reward_1.sum(dim=-1)
#             r2 = reward_2.sum(dim=-1)

#             logits = r2 - r1
#             labels = batch["label"].float().unsqueeze(0).expand_as(logits)

#             assert labels.shape == logits.shape
#             reward_loss = self.reward_criterion(logits, labels).mean()

#             with torch.no_grad():
#                 reward_accuracy = ((r2 > r1) == torch.round(labels)).float().mean()

#             self.optim["reward"].zero_grad(set_to_none=True)
#             reward_loss.backward()
#             self.optim["reward"].step()

#             # reward = reward.detach().mean(dim=0)
#             reward = reward.detach().mean(dim=0)
#         else:
#             with torch.no_grad():
#                 # reward = self.network.reward(obs, action).mean(dim=0)
#                 # traj_feature = self.network.trajectory_encoder(obs, action)
#                 # task_feature = self.network.task_encoder(task_tokens, task_masks)
#                 # reward = traj_feature * task_feature
#                 reward_1 = self.network.reward(obs_1, action_1, task_tokens, task_masks)
#                 reward_2 = self.network.reward(obs_2, action_2, task_tokens, task_masks)
#                 reward = torch.cat([reward_1, reward_2], dim=1).mean(dim=0)

#         # # Encode everything
#         # obs = self.network.encoder(obs)
#         # next_obs = obs[:, 1:].detach()
#         # obs = obs[:, :-1]
#         # action = action[:, :-1]
#         # if isinstance(discount, torch.Tensor):
#         #     discount = discount[:, :-1]
#         # reward = reward[:, :-1]

#         # with torch.no_grad():
#         #     target_q = self.target_network.critic(obs, action)
#         #     target_q = torch.min(target_q, dim=0)[0]
#         # vs = self.network.value(obs.detach())
#         # v_loss = iql_loss(vs, target_q.unsqueeze(0).expand_as(vs), self.expectile).mean()

#         # self.optim["value"].zero_grad(set_to_none=True)
#         # v_loss.backward()
#         # self.optim["value"].step()

#         # # Next, update the actor. We detach and use the old value, v for computational efficiency
#         # # and use the target_q value though the JAX IQL recomputes both
#         # # Pytorch IQL versions have not.
#         # with torch.no_grad():
#         #     adv = target_q - torch.mean(vs, dim=0)  # min trick is not used on value.
#         #     exp_adv = torch.exp(adv / self.beta)
#         #     if self.clip_score is not None:
#         #         exp_adv = torch.clamp(exp_adv, max=self.clip_score)

#         # dist = self.network.actor(obs)  # Use encoder gradients for the actor.
#         # if isinstance(dist, torch.distributions.Distribution):
#         #     bc_loss = -dist.log_prob(action)
#         # elif torch.is_tensor(dist):
#         #     assert dist.shape == action.shape
#         #     bc_loss = torch.nn.functional.mse_loss(dist, action, reduction="none").sum(dim=-1)
#         # else:
#         #     raise ValueError("Invalid policy output provided")
#         # actor_loss = (exp_adv * bc_loss).mean()

#         # self.optim["actor"].zero_grad(set_to_none=True)
#         # actor_loss.backward()
#         # self.optim["actor"].step()

#         # # Next, Finally update the critic
#         # with torch.no_grad():
#         #     next_vs = self.network.value(next_obs)
#         #     next_v = torch.mean(next_vs, dim=0, keepdim=True)  # Min trick is not used on value.
#         #     target = reward + discount * next_v  # use the predicted reward.
#         # qs = self.network.critic(obs.detach(), action)
#         # q_loss = torch.nn.functional.mse_loss(qs, target.expand_as(qs), reduction="none").mean()

#         # self.optim["critic"].zero_grad(set_to_none=True)
#         # q_loss.backward()
#         # self.optim["critic"].step()

#         # metrics = dict(
#         #     q_loss=q_loss.item(),
#         #     v_loss=v_loss.item(),
#         #     actor_loss=actor_loss.item(),
#         #     v=vs.mean().item(),
#         #     q=qs.mean().item(),
#         #     adv=adv.mean().item(),
#         #     reward=reward.mean().item(),
#         # )

#         reward = reward[:, :-1]
#         metrics = dict(
#             reward=reward.mean().item(),
#         )

#         if step < self.reward_steps:
#             metrics["reward_loss"] = reward_loss.item()
#             metrics["reward_accuracy"] = reward_accuracy.item()

#         # Update the networks. These are done in a stack to support different grad options for the encoder.
#         # if step % self.target_freq == 0:
#         #     with torch.no_grad():
#         #         # Only run on the critic and encoder, those are the only weights we update.
#         #         for param, target_param in zip(
#         #             self.network.critic.parameters(), self.target_network.critic.parameters()
#         #         ):
#         #             target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

#         return metrics

#     def validation_step(self, batch: Dict) -> Dict:
#         # Compute the loss
#         print("validation called")
#         metrics = {}
#         with torch.no_grad():
#             obs_1 = batch["obs_1"]
#             obs_2 = batch["obs_2"]

#             action_1 = batch["action_1"]
#             action_2 = batch["action_2"]

#             task_tokens = batch["task_tokens"]
#             task_masks = batch["task_masks"]
#             task_labels = batch["task_labels"]

#             for i in self.eval_tasks:
#                 reward_t1 = self.network.reward(obs_1[task_labels==i], action_1[task_labels==i], task_tokens[task_labels==i], task_masks[task_labels==i])
#                 reward_t2 = self.network.reward(obs_2[task_labels==i], action_2[task_labels==i], task_tokens[task_labels==i], task_masks[task_labels==i])
#                 r_t1 = reward_t1.sum(dim=-1)
#                 r_t2 = reward_t2.sum(dim=-1)
#                 labels = batch["label"][task_labels==i].float().unsqueeze(0).expand_as(r_t1)
#                 task_accuracy = ((r_t2 > r_t1) == torch.round(labels)).float().mean()
#                 metrics[f"task_{i}_accuracy"] = task_accuracy.item()

#         return metrics

#     def _get_train_action(self, obs: Any, step: int, total_steps: int):
#         batch = dict(obs=obs)
#         with torch.no_grad():
#             action = self.predict(batch, is_batched=False, sample=True)
#         return action

class MTPIQL(OffPolicyAlgorithm):
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
        task_embeddings = torch.cat([batch["task_embeddings"], batch["task_embeddings"]], dim=0)  # (B, S+1)

        if step < self.reward_steps:
            self.network.reward.train()
            traj_embeddings = self.network.reward(obs, action)
            reward = torch.einsum('obtd,bd->obt', traj_embeddings, task_embeddings)

            r1, r2 = torch.chunk(reward.sum(dim=-1), 2, dim=1)  # Should return two (E, B)
            logits = r2 - r1
            labels = batch["label"].float().unsqueeze(0).expand_as(logits)

            assert labels.shape == logits.shape
            reward_loss = self.reward_criterion(logits, labels).mean()

            with torch.no_grad():
                reward_accuracy = ((r2 > r1) == torch.round(labels)).float().mean()

            self.optim["reward"].zero_grad(set_to_none=True)
            reward_loss.backward()
            self.optim["reward"].step()

            reward = reward.detach().mean(dim=0)
        else:
            with torch.no_grad():
                traj_embeddings = self.network.reward(obs, action)
                reward = torch.einsum('obtd,bd->obt', traj_embeddings, task_embeddings).mean(dim=0)

        # # Encode everything
        # obs = self.network.encoder(obs)
        # next_obs = obs[:, 1:].detach()
        # obs = obs[:, :-1]
        # action = action[:, :-1]
        # # discount = discount[:, :-1]
        # reward = reward[:, :-1]
        # traj_dim = reward.shape[-1]
        # task_embeddings = task_embeddings.unsqueeze(1).expand(-1, traj_dim, -1)  # (B, S, T)

        # with torch.no_grad():
        #     target_q = self.target_network.critic(obs, action, task_embeddings)
        #     target_q = torch.min(target_q, dim=0)[0]
        # vs = self.network.value(obs.detach(), task_embeddings.detach())
        # v_loss = iql_loss(vs, target_q.unsqueeze(0).expand_as(vs), self.expectile).mean()

        # self.optim["value"].zero_grad(set_to_none=True)
        # v_loss.backward()
        # self.optim["value"].step()

        # # Next, update the actor. We detach and use the old value, v for computational efficiency
        # # and use the target_q value though the JAX IQL recomputes both
        # # Pytorch IQL versions have not.
        # with torch.no_grad():
        #     adv = target_q - torch.mean(vs, dim=0)  # min trick is not used on value.
        #     exp_adv = torch.exp(adv / self.beta)
        #     if self.clip_score is not None:
        #         exp_adv = torch.clamp(exp_adv, max=self.clip_score)

        # dist = self.network.actor(obs, task_embeddings)  # Use encoder gradients for the actor.
        # if isinstance(dist, torch.distributions.Distribution):
        #     bc_loss = -dist.log_prob(action)
        # elif torch.is_tensor(dist):
        #     assert dist.shape == action.shape
        #     bc_loss = torch.nn.functional.mse_loss(dist, action, reduction="none").sum(dim=-1)
        # else:
        #     raise ValueError("Invalid policy output provided")
        # actor_loss = (exp_adv * bc_loss).mean()

        # self.optim["actor"].zero_grad(set_to_none=True)
        # actor_loss.backward()
        # self.optim["actor"].step()

        # # Next, Finally update the critic
        # with torch.no_grad():
        #     next_vs = self.network.value(next_obs, task_embeddings)
        #     next_v = torch.mean(next_vs, dim=0, keepdim=True)  # Min trick is not used on value.
        #     target = reward + self.discount * next_v  # use the predicted reward.
        # qs = self.network.critic(obs.detach(), action, task_embeddings.detach())
        # q_loss = torch.nn.functional.mse_loss(qs, target.expand_as(qs), reduction="none").mean()

        # self.optim["critic"].zero_grad(set_to_none=True)
        # q_loss.backward()
        # self.optim["critic"].step()

        # metrics = dict(
        #     q_loss=q_loss.item(),
        #     v_loss=v_loss.item(),
        #     actor_loss=actor_loss.item(),
        #     v=vs.mean().item(),
        #     q=qs.mean().item(),
        #     adv=adv.mean().item(),
        #     reward=reward.mean().item(),
        # )
        
        metrics = dict()
        if step < self.reward_steps:
            metrics["reward_loss"] = reward_loss.item()
            metrics["reward_accuracy"] = reward_accuracy.item()

        # Update the networks. These are done in a stack to support different grad options for the encoder.
        if step % self.target_freq == 0:
            with torch.no_grad():
                # Only run on the critic and encoder, those are the only weights we update.
                for param, target_param in zip(
                    self.network.critic.parameters(), self.target_network.critic.parameters()
                ):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return metrics

    # def validation_step(self, batch: Dict) -> Dict:
    #     # Compute the loss
    #     if isinstance(batch, (tuple, list)) and "label" in batch[1]:
    #         feedback_batch = batch[1]
    #         with torch.no_grad():
    #             reward_loss, reward_accuracy, reward_pred = self._get_reward_loss(feedback_batch)
    #         return dict(
    #             reward_loss=reward_loss.item(), reward_accuracy=reward_accuracy.item(), reward=reward_pred.mean().item()
    #         )
    #     else:
    #         return dict()
        
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
    
    def predict_reward(self, batch: Dict, obs_key="obs", action_key="action", task_embedding_key="task_embeddings") -> torch.Tensor:
        """
        Predict the reward for the given batch.
        """
        with torch.no_grad():
            obs = batch[obs_key]
            action = batch[action_key]
            task_embeddings = batch[task_embedding_key]
            traj_embeddings = self.network.reward(obs, action)
            reward = torch.einsum('obt,bt->b', traj_embeddings, task_embeddings)
        return reward

    def _get_train_action(self, obs: Any, step: int, total_steps: int):
        batch = dict(obs=obs)
        with torch.no_grad():
            action = self.predict(batch, is_batched=False, sample=True)
        return action
