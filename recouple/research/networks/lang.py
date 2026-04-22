
import torch
from torch import nn
from transformers import T5EncoderModel

HF_LANG_MODEL_NAME = {
    # BERT models
    'bert-base': 'google/bert_uncased_L-12_H-768_A-12',
    'bert-mini': 'google/bert_uncased_L-4_H-256_A-4',
    'bert-tiny': 'google/bert_uncased_L-4_H-128_A-2',

    # T5 models
    't5-small': 'google-t5/t5-small',
    't5-base': 'google-t5/t5-base',
}

class LanguageEncoder(nn.Module):
    def __init__(
        self,
        lang_model_name="t5-base", #"t5-small"
        finetune=False,
    ):
        self.lang_encoder = T5EncoderModel.from_pretrained(HF_LANG_MODEL_NAME[lang_model_name])

    def forward(self, tokens, masks) -> torch.Tensor:
        bert_outputs = self.lang_encoder(tokens, attention_mask=masks)
        bert_embeddings = bert_outputs.last_hidden_state
        encoded_lang = torch.mean(bert_embeddings, dim=1, keepdim=False)
        
        return encoded_lang