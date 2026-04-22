from collections.abc import Iterable
from typing import List
import gym
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer, T5Config, T5EncoderModel

from .mlp import MLP, EnsembleMLP

HF_LANG_MODEL_NAME = {
    # BERT models
    'bert-base': 'google/bert_uncased_L-12_H-768_A-12',
    'bert-mini': 'google/bert_uncased_L-4_H-256_A-4',
    'bert-tiny': 'google/bert_uncased_L-4_H-128_A-2',

    # T5 models
    't5-small': 'google-t5/t5-small',
    't5-base': 'google-t5/t5-base',
}

LANG_EMBED_DIMS = {
    't5-small': 512,
    't5-base': 768,
}


def drqv2_weight_init(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


class DrQv2Encoder(nn.Module):
    def __init__(self, observation_space: gym.Space, action_space: gym.Space) -> None:
        super().__init__()
        if len(observation_space.shape) == 4:
            s, c, h, w = observation_space.shape
            channels = s * c
        elif len(observation_space.shape) == 3:
            c, h, w = observation_space.shape
            channels = c
        else:
            raise ValueError("Invalid observation space for DRQV2 Image encoder.")
        self.convnet = nn.Sequential(
            nn.Conv2d(channels, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.reset_parameters()

        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]) / 255.0 - 0.5
            self.repr_dim = self.convnet(sample).shape[1]

    def reset_parameters(self):
        self.apply(drqv2_weight_init)

    @property
    def output_space(self) -> gym.Space:
        return gym.spaces.Box(shape=(self.repr_dim,), low=-np.inf, high=np.inf, dtype=np.float32)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        is_seq = len(obs.shape) == 5
        if is_seq:
            b, s, c, h, w = obs.shape
            obs = obs.view(b * s, c, h, w)
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        if is_seq:
            h = h.view(b, s, self.repr_dim)
        return h


class DrQv2Critic(nn.Module):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        feature_dim: int = 50,
        hidden_layers: List[int] = (1024, 1024),
        ensemble_size: int = 2,
        **kwargs,
    ):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(observation_space.shape[0], feature_dim), nn.LayerNorm(feature_dim), nn.Tanh()
        )
        self.ensemble_size = ensemble_size
        input_dim = feature_dim + action_space.shape[0]
        if self.ensemble_size > 1:
            self.mlp = EnsembleMLP(input_dim, 1, ensemble_size=ensemble_size, hidden_layers=hidden_layers, **kwargs)
        else:
            self.mlp = MLP(input_dim, 1, hidden_layers=hidden_layers, **kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        self.apply(drqv2_weight_init)

    def forward(self, obs, action):
        x = self.trunk(obs)
        x = torch.cat((x, action), dim=-1)
        q = self.mlp(x).squeeze(-1)  # Remove the last dim
        if self.ensemble_size == 1:
            q = q.unsqueeze(0)  # add in the ensemble dim
        return q


class DrQv2Value(nn.Module):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        feature_dim: int = 50,
        hidden_layers: List[int] = (1024, 1024),
        ensemble_size: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(observation_space.shape[0], feature_dim), nn.LayerNorm(feature_dim), nn.Tanh()
        )
        self.ensemble_size = ensemble_size
        if self.ensemble_size > 1:
            self.mlp = EnsembleMLP(feature_dim, 1, ensemble_size=ensemble_size, hidden_layers=hidden_layers, **kwargs)
        else:
            self.mlp = MLP(feature_dim, 1, hidden_layers=hidden_layers, **kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        self.apply(drqv2_weight_init)

    def forward(self, obs):
        v = self.trunk(obs)
        v = self.mlp(v).squeeze(-1)  # Remove the last dim
        if self.ensemble_size == 1:
            v = v.unsqueeze(0)  # add in the ensemble dim
        return v


class DrQv2Actor(nn.Module):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        feature_dim: int = 50,
        hidden_layers: List[int] = (1024, 1024),
        **kwargs,
    ):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(observation_space.shape[0], feature_dim), nn.LayerNorm(feature_dim), nn.Tanh()
        )
        self.mlp = MLP(feature_dim, action_space.shape[0], hidden_layers=hidden_layers, **kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        self.apply(drqv2_weight_init)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.trunk(obs)
        return self.mlp(x)

class DrQv2Reward(nn.Module):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        feature_dim: int = 100,
        hidden_layers: List[int] = (512, 512),
        ensemble_size: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.encoder = DrQv2Encoder(observation_space, action_space)

        self.trunk = nn.Sequential(nn.Linear(self.encoder.repr_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh())
        self.ensemble_size = ensemble_size
        input_dim = feature_dim + action_space.shape[0]
        if self.ensemble_size > 1:
            self.mlp = EnsembleMLP(input_dim, 1, ensemble_size=ensemble_size, hidden_layers=hidden_layers, **kwargs)
        else:
            self.mlp = MLP(input_dim, 1, hidden_layers=hidden_layers, **kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        self.apply(drqv2_weight_init)

    def forward(self, obs, action):
        x = self.encoder(obs)
        x = self.trunk(x)
        x = torch.cat((x, action), dim=-1)
        q = self.mlp(x).squeeze(-1)  # Remove the last dim
        if self.ensemble_size == 1:
            q = q.unsqueeze(0)  # add in the ensemble dim
        return q
    
class DrQv2MTReward(nn.Module):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        feature_dim: int = 100,
        hidden_layers: List[int] = (512, 512),
        ensemble_size: int = 1,
        lang_model_name="t5-small",
        randomize_lang_encoder=False,
        freeze_lang_encoder=False,
        use_cosine_similarity=False,
        **kwargs,
    ):
        super().__init__()
        self.traj_encoder = DrQv2Encoder(observation_space, action_space)
        self.trunk = nn.Sequential(nn.Linear(self.traj_encoder.repr_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh())
        self.ensemble_size = ensemble_size
        input_dim = feature_dim + action_space.shape[0]
        lang_embed_dim = LANG_EMBED_DIMS[lang_model_name]
        self.mlp = MLP(input_dim, lang_embed_dim, hidden_layers=hidden_layers, **kwargs)
        self.reset_parameters()

        t5_config = T5Config.from_pretrained(lang_model_name)
        t5_config.dropout_rate = 0.0
        if randomize_lang_encoder:
            assert not freeze_lang_encoder
            self.lang_encoder = T5EncoderModel(t5_config)
        else:
            self.lang_encoder = T5EncoderModel.from_pretrained(
                HF_LANG_MODEL_NAME[lang_model_name],
                config=t5_config,
            )
        if freeze_lang_encoder:
            for param in self.lang_encoder.parameters():
                param.requires_grad = False

        self.use_cosine_similarity = use_cosine_similarity

    def reset_parameters(self):
        self.apply(drqv2_weight_init)

    def get_traj_embedding(self, obs, action):
        x = self.traj_encoder(obs)
        x = self.trunk(x)
        x = torch.cat((x, action), dim=-1)
        traj_embedding = self.mlp(x)
        return traj_embedding

    def get_task_embedding(self, tokens, masks):
        task_bert_outputs = self.lang_encoder(tokens, attention_mask=masks)
        task_bert_embeddings = task_bert_outputs.last_hidden_state
        task_embedding = torch.mean(task_bert_embeddings, dim=1, keepdim=False)
        return task_embedding

    def forward(self, obs, action, tokens, masks):
        traj_embedding = self.get_traj_embedding(obs, action)

        task_embedding = self.get_task_embedding(tokens, masks)
        task_embedding = task_embedding.unsqueeze(-1)

        if self.use_cosine_similarity:
            reward = torch.bmm(
                F.normalize(traj_embedding, dim=-1), 
                F.normalize(task_embedding, dim=-1)
            )
        else:
            reward = torch.bmm(traj_embedding, task_embedding)

        reward = reward.squeeze(-1)
        if self.ensemble_size == 1:
            reward = reward.unsqueeze(0)  # add in the ensemble dim
        return reward

class DrQv2MTReasonedReward(nn.Module):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        feature_dim: int = 100,
        hidden_layers: List[int] = (512, 512),
        ensemble_size: int = 1,
        lang_model_name="t5-small",
        randomize_lang_encoder=False,
        freeze_task_encoder=False,
        freeze_reason_encoder=False,
        use_cosine_similarity=False,
        **kwargs,
    ):
        super().__init__()
        self.traj_encoder = DrQv2Encoder(observation_space, action_space)
        self.trunk = nn.Sequential(nn.Linear(self.traj_encoder.repr_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh())
        self.ensemble_size = ensemble_size
        input_dim = feature_dim + action_space.shape[0]
        lang_embed_dim = LANG_EMBED_DIMS[lang_model_name]
        self.mlp = MLP(input_dim, lang_embed_dim, hidden_layers=hidden_layers, **kwargs)
        self.reset_parameters()

        t5_config = T5Config.from_pretrained(lang_model_name)
        t5_config.dropout_rate = 0.0
        if randomize_lang_encoder:
            assert not freeze_task_encoder and not freeze_reason_encoder
            self.task_encoder = T5EncoderModel(t5_config)
            self.reason_encoder = T5EncoderModel(t5_config)
        else:
            self.task_encoder = T5EncoderModel.from_pretrained(
                HF_LANG_MODEL_NAME[lang_model_name],
                config=t5_config,
            )
            self.reason_encoder = T5EncoderModel.from_pretrained(
                HF_LANG_MODEL_NAME[lang_model_name],
                config=t5_config,
            )
        if freeze_task_encoder:
            for param in self.task_encoder.parameters():
                param.requires_grad = False
        if freeze_reason_encoder:
            for param in self.reason_encoder.parameters():
                param.requires_grad = False
        
        self.use_cosine_similarity = use_cosine_similarity

    def reset_parameters(self):
        self.apply(drqv2_weight_init)

    def get_traj_embedding(self, obs, action):
        x = self.traj_encoder(obs)
        x = self.trunk(x)
        x = torch.cat((x, action), dim=-1)
        traj_embedding = self.mlp(x)
        return traj_embedding
    
    def get_reason_embedding(self, reason_tokens, reason_masks):
        reason_bert_outputs = self.reason_encoder(reason_tokens, attention_mask=reason_masks)
        reason_bert_embeddings = reason_bert_outputs.last_hidden_state
        reason_embedding = torch.mean(reason_bert_embeddings, dim=1, keepdim=False)
        return reason_embedding
    
    def get_task_embedding(self, task_tokens, task_masks):
        task_bert_outputs = self.task_encoder(task_tokens, attention_mask=task_masks)
        task_bert_embeddings = task_bert_outputs.last_hidden_state
        task_embedding = torch.mean(task_bert_embeddings, dim=1, keepdim=False)
        return task_embedding

    def get_reason_value(self, obs, action, reason_tokens, reason_masks):
        traj_embedding = self.get_traj_embedding(obs, action)

        reason_embedding = self.get_reason_embedding(reason_tokens, reason_masks)
        reason_embedding = reason_embedding.unsqueeze(-1)

        if self.use_cosine_similarity:
            reason_value = torch.bmm(
                F.normalize(traj_embedding, dim=-1), 
                F.normalize(reason_embedding, dim=-1)
            )
        else:
            reason_value = torch.bmm(traj_embedding, reason_embedding)
        
        reason_value = reason_value.squeeze(-1)
        if self.ensemble_size == 1:
            reason_value = reason_value.unsqueeze(0)  # add in the ensemble dim
        return reason_value
    
    def get_reason_task_alignment(self, reason_tokens, reason_masks, task_tokens, task_masks):
        reason_embedding = self.get_reason_embedding(reason_tokens, reason_masks)
        task_embedding = self.get_task_embedding(task_tokens, task_masks)
        if self.use_cosine_similarity:
            alignment_value = torch.einsum(
                "bd,bd->b", 
                F.normalize(reason_embedding, dim=-1), 
                F.normalize(task_embedding, dim=-1)
            )
        else:
            alignment_value = torch.einsum("bd,bd->b", reason_embedding, task_embedding)
        return alignment_value

    def forward(self, obs, action, tokens, masks):
        traj_embedding = self.get_traj_embedding(obs, action)

        task_embedding = self.get_task_embedding(tokens, masks)
        task_embedding = task_embedding.unsqueeze(-1)

        if self.use_cosine_similarity:
            reward = torch.bmm(
                F.normalize(traj_embedding, dim=-1), 
                F.normalize(task_embedding, dim=-1)
            )
        else:
            reward = torch.bmm(traj_embedding, task_embedding)

        reward = reward.squeeze(-1)
        if self.ensemble_size == 1:
            reward = reward.unsqueeze(0)  # add in the ensemble dim
        return reward

class DrQv2MTReasonedRewardNew(nn.Module):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        feature_dim: int = 100,
        hidden_layers: List[int] = (512, 512),
        ensemble_size: int = 1,
        lang_model_name="t5-small",
        randomize_lang_encoder=False,
        freeze_lang_encoder=False,
        **kwargs,
    ):
        super().__init__()
        self.traj_encoder = DrQv2Encoder(observation_space, action_space)
        self.trunk = nn.Sequential(nn.Linear(self.traj_encoder.repr_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh())
        self.ensemble_size = ensemble_size
        input_dim = feature_dim + action_space.shape[0]
        lang_embed_dim = LANG_EMBED_DIMS[lang_model_name]
        self.mlp = MLP(input_dim, lang_embed_dim, hidden_layers=hidden_layers, **kwargs)
        self.reset_parameters()

        t5_config = T5Config.from_pretrained(lang_model_name)
        t5_config.dropout_rate = 0.0
        if randomize_lang_encoder:
            assert not freeze_lang_encoder
            self.lang_encoder = T5EncoderModel(t5_config)
        else:
            self.lang_encoder = T5EncoderModel.from_pretrained(
                HF_LANG_MODEL_NAME[lang_model_name],
                config=t5_config,
            )
        if freeze_lang_encoder:
            for param in self.lang_encoder.parameters():
                param.requires_grad = False

    def reset_parameters(self):
        self.apply(drqv2_weight_init)

    def get_traj_embedding(self, obs, action):
        x = self.traj_encoder(obs)
        x = self.trunk(x)
        x = torch.cat((x, action), dim=-1)
        traj_embedding = self.mlp(x)
        traj_embedding = F.normalize(traj_embedding, dim=-1)
        return traj_embedding
    
    def get_reason_embedding(self, reason_tokens, reason_masks):
        reason_bert_outputs = self.lang_encoder(reason_tokens, attention_mask=reason_masks)
        reason_bert_embeddings = reason_bert_outputs.last_hidden_state
        reason_embedding = torch.mean(reason_bert_embeddings, dim=1, keepdim=False)
        reason_embedding = F.normalize(reason_embedding, dim=-1)
        return reason_embedding
    
    def get_task_embedding(self, task_tokens, task_masks):
        task_bert_outputs = self.lang_encoder(task_tokens, attention_mask=task_masks)
        task_bert_embeddings = task_bert_outputs.last_hidden_state
        task_embedding = torch.mean(task_bert_embeddings, dim=1, keepdim=False)
        task_embedding = F.normalize(task_embedding, dim=-1)
        return task_embedding
    
    def decompose_traj_embedding(self, traj_embedding, reason_embedding):
        reason_embedding = reason_embedding.unsqueeze(1)  # (B, 1, D)
        dot_product = (traj_embedding * reason_embedding).sum(dim=-1, keepdim=True)  # (B, T, 1)
        reason_aligned = dot_product * reason_embedding  # (B, T, D)
        reason_orthogonal = traj_embedding - reason_aligned
        return reason_aligned, reason_orthogonal

    def get_reason_values(self, obs, action, reason_tokens, reason_masks, task_tokens, task_masks):
        traj_embedding = self.get_traj_embedding(obs, action)
        reason_embedding = self.get_reason_embedding(reason_tokens, reason_masks)
        task_embedding = self.get_task_embedding(task_tokens, task_masks)
        reason_aligned, reason_orthogonal = self.decompose_traj_embedding(traj_embedding, reason_embedding)
        
        # reason_embedding = reason_embedding.unsqueeze(-1)
        task_embedding = task_embedding.unsqueeze(-1)

        reason_aligned_value = torch.bmm(reason_aligned, task_embedding).squeeze(-1)
        reason_orthogonal_value = torch.bmm(reason_orthogonal, task_embedding).squeeze(-1)
        
        if self.ensemble_size == 1:
            reason_aligned_value = reason_aligned_value.unsqueeze(0)  # add in the ensemble dim
            reason_orthogonal_value = reason_orthogonal_value.unsqueeze(0)  # add in the ensemble dim
        
        # return reason_aligned_value, reason_orthogonal_value
        return reason_aligned_value, reason_orthogonal_value
    
    def forward(self, obs, action, tokens, masks):
        traj_embedding = self.get_traj_embedding(obs, action)
        task_embedding = self.get_task_embedding(tokens, masks)
        task_embedding = task_embedding.unsqueeze(-1)

        reward = torch.bmm(traj_embedding, task_embedding)

        reward = reward.squeeze(-1)
        if self.ensemble_size == 1:
            reward = reward.unsqueeze(0)  # add in the ensemble dim
        return reward
