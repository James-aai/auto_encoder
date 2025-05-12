import torch
import torch.nn as nn
from transformers import T5Tokenizer, T5ForConditionalGeneration

class SparseT5Autoencoder(nn.Module):
    def __init__(self, rank, sequence_token_length, latent_dim, sparsity_k):
        super().__init__()
        self.rank = rank
        self.latent_dim = latent_dim
        self.sparsity_k = sparsity_k

        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
        self.base_model = T5ForConditionalGeneration.from_pretrained('t5-small')

        self.encoder_projection = nn.Linear(self.base_model.config.d_model, latent_dim)
        self.decoder_projection = nn.Linear(latent_dim, self.base_model.config.d_model)

    def top_k_sparsify(self, x, k):
        values, indices = torch.topk(x.abs(), k, dim=-1)
        mask = torch.zeros_like(x).scatter_(-1, indices, 1.0)
        return x * mask

    def forward(self, input_text):
        input_text_prefixed = "reconstruct: " + input_text
        inputs = self.tokenizer(input_text_prefixed, return_tensors='pt', padding='max_length',
                                truncation=True, max_length=self.sequence_token_length)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        target = self.tokenizer(input_text, return_tensors='pt', padding='max_length',
                                truncation=True, max_length=self.sequence_token_length)
        labels = target['input_ids']
        labels[labels == self.tokenizer.pad_token_id] = -100

        encoder_outputs = self.base_model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = encoder_outputs.last_hidden_state

        latent = self.encoder_projection(hidden_states)
        sparse_latent = self.top_k_sparsify(latent, self.sparsity_k)
        dense_reconstruction = self.decoder_projection(sparse_latent)

        outputs = self.base_model(
            encoder_outputs=(dense_reconstruction,),
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs.loss, outputs.logits
