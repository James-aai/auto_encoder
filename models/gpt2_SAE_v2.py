import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
from tqdm import tqdm
import torch.nn.init as init
model_id = "openai-community/gpt2"
from sparsemax import Sparsemax

def load_tokenizer():
    print("----------Loading tokenizer----------")
    tokenizer = GPT2Tokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

class SparseAutoEnc(nn.Module):
    def __init__(self, rank, sequence_token_length, latent_dim, sparsity_k, gpt2_model='gpt2'):
        super().__init__()

        # Set device based on rank
        self.device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
        self.sequence_token_length = sequence_token_length

        # Load encoder and decoder onto the specified device
        self.encoder = GPT2Model.from_pretrained(gpt2_model).to(self.device)
        self.decoder = GPT2LMHeadModel.from_pretrained(gpt2_model).to(self.device)

        # Freeze encoder parameters
        # for param in self.encoder.parameters():
        #     param.requires_grad = False

        # Freeze encoder parameters
        # for param in self.decoder.parameters():
        #     param.requires_grad = False

        # Latent projection layers
        self.latent_projector = nn.Linear(self.encoder.config.n_embd * self.sequence_token_length, latent_dim).to(self.device)
        self.init_linear(self.latent_projector)

        self.reverse_projector = nn.Linear(latent_dim, self.decoder.config.n_embd * self.sequence_token_length).to(self.device)
        self.init_linear(self.reverse_projector)

        self.sparsemax = Sparsemax( dim=-1)

        self.sparsity_k = sparsity_k
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100).to(self.device)

    def init_linear(self, linear):
        init.kaiming_normal_(linear.weight, nonlinearity='relu')
        if linear.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(linear.weight)
            bound = 1 / fan_in ** 0.5
            init.uniform_(linear.bias, -bound, bound)


    def topk_sparsify(self, x, k):
        """ Keep top-k activations in each row, zero the rest """
        topk_vals, _ = torch.topk(x, k=k, dim=-1)
        threshold = topk_vals[..., -1, None]
        return x * (x >= threshold)

    def forward(self, input_ids, attention_mask=None, labels=None):
        # print("input ids:", input_ids)
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = encoder_outputs.last_hidden_state  # [B, T, D]
        # print("Encoder hidden states:", hidden_states)

        # latent_repr = hidden_states.mean(dim=1)  # [B, D]
        projected = self.latent_projector(hidden_states.view(hidden_states.shape[0], -1))  # [B, latent_dim]
        # projected_activ = self.activ
        # Apply top-k sparsity
        sparse_latent = self.topk_sparsify(projected, self.sparsity_k)
        # sparse_latent = self.sparsemax(projected)

        # Reproject to GPT-2 hidden size
        reprojected = self.reverse_projector(sparse_latent)  # [B, D]
        # print("Reprojected: ", reprojected)

        # Expand to sequence length for decoding
        # expanded = reprojected.unsqueeze(1).repeat(1, input_ids.size(1), 1)  # [B, T, D]
        expanded = reprojected.view(-1, self.sequence_token_length, self.decoder.config.n_embd)
        # print("Expanded: ", expanded)

        # Decode using GPT-2
        outputs = self.decoder(inputs_embeds=expanded)
        logits = outputs.logits

        # Calculate loss manually
        if labels is None:
            return logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return loss  # , logits, sparse_latent
