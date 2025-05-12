import torch
import torch.nn as nn
from transformers import BartTokenizer, BartForConditionalGeneration

model_id = "facebook/bart-base"

def load_tokenizer():
    print("----------Loading tokenizer----------")
    tokenizer = BartTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

class SparseAutoEnc(nn.Module):
    def __init__(self, rank, sequence_token_length, latent_dim, sparsity_k):
        super().__init__()
        self.rank = rank
        self.latent_dim = latent_dim
        self.sparsity_k = sparsity_k

        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        self.base_model = BartForConditionalGeneration.from_pretrained('facebook/bart-base').to(self.rank)

        # Project encoder hidden states down to latent space
        self.encoder_projection = nn.Linear(self.base_model.config.d_model, latent_dim).to(self.rank)

        # Project back up to original hidden size
        self.decoder_projection = nn.Linear(latent_dim, self.base_model.config.d_model).to(self.rank)

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100).to(self.rank)

    def top_k_sparsify(self, x, k):
        values, indices = torch.topk(x.abs(), k, dim=-1)
        mask = torch.zeros_like(x).scatter_(-1, indices, 1.0).to(self.rank)
        return x * mask

    def forward(self, input_ids, attention_mask=None, labels=None):

        encoder_outputs = self.base_model.model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = encoder_outputs.last_hidden_state  # [B, T, H]

        latent = self.encoder_projection(hidden_states)
        sparse_latent = self.top_k_sparsify(latent, self.sparsity_k)
        dense_reconstruction = self.decoder_projection(sparse_latent)

        outputs = self.base_model(
            encoder_outputs=(dense_reconstruction,),
            attention_mask=attention_mask,
            labels=input_ids
        )

        logits = outputs.logits

        # Calculate loss manually
        if labels is None:
            return logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return loss  # , logits, sparse_latent
