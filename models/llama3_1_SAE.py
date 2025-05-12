import torch

# model_id = "facebook/opt-125m"
# model_id = "facebook/opt-350m"
# model_id = "facebook/opt-6.7b"
# model_id = "facebook/opt-2.7b"
# model_id = "meta-llama/Meta-Llama-3.1-8B"
model_id = "meta-llama/Llama-3.2-3B"
# model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
# model_id = '/scratch/dg97/jm6832/Meta-Llama-3.1-8B/' # for local model
# model_id = "/scratch/dg97/jm6832/hf_home/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/48d6d0fc4e02fb1269b36940650a1b7233035cbb"
# model_id = "openai-community/gpt2"
local_files_only=False

# useful references:
# https://github.com/graykode/gpt-2-Pytorch/blob/master/GPT2/model.py
# https://stackoverflow.com/questions/73593136/typeerror-dropout-argument-input-position-1-must-be-tensor-not-tuple
def load_tokenizer():
    print("----------Loading tokenizer----------")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, padding_side='right',
                                              local_files_only=local_files_only)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

class SparseAutoEnc(torch.nn.Module):
    def __init__(self, rank, sequence_token_length, embed_dim, sparsity_k):
        super(SparseAutoEnc, self).__init__()
        self.rank = rank
        self.k = sparsity_k
        self.logit_dim = 3072  # 768 for GPT-2, 3072 for Llama 3.2
        self.sparse_act = torch.nn.ReLU()
        self.device_main = torch.device(f"cuda:{rank}")
        self.device_aux = torch.device("cuda:1")
        # self.device_aux = self.device_main

        enc_model = self._load_model(rank)
        self.encoder_0 = list(enc_model.children())[0].to(self.device_main)

        self.embed_0 = torch.nn.Linear(
            in_features=self.logit_dim * sequence_token_length, out_features=embed_dim, dtype=torch.float16
        ).to(self.device_main)

        self.embed_1 = torch.nn.Linear(
            in_features=embed_dim, out_features=self.logit_dim * sequence_token_length, dtype=torch.float16
        ).to(self.device_aux)

        dec_model = self._load_model(self.device_aux)
        decoder_embedding_layers_to_drop = 1
        self.decoder_0_list = list(list(dec_model.children())[0].children())[decoder_embedding_layers_to_drop].to(
            self.device_aux)
        self.decoder_1 = list(list(dec_model.children())[0].children())[2].to(self.device_aux)
        self.decoder_3 = list(dec_model.children())[1].to(self.device_aux)
        self.rotary_emb = dec_model.model.rotary_emb
        self.norm = dec_model.model.norm

    def forward(self, input_ids, labels=None, **kwargs):
        output = self.encoder_0(input_ids.to(self.device_main))[0]
        local_batch_size, sequence_length = input_ids.shape
        output = output.view(local_batch_size, -1)
        output = self.embed_0(output).to(self.device_main)

        if self.device_aux != self.device_main:
            output = output.cpu().to(self.device_aux)

        output = self._sparsify(output)
        output = self.embed_1(output) ### why is this producing NaN ???
        output = output.view(-1, sequence_length, self.logit_dim)

        position_ids = torch.arange(0, output.shape[1], device=self.device_aux).unsqueeze(0)
        position_embeddings = self.rotary_emb(output, position_ids)

        for block in self.decoder_0_list:
            output = block(output, position_embeddings=position_embeddings)[0]

        output = self.norm(output)
        output = self.decoder_3(output)

        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)
            labels = labels.view(-1)
            if self.device_aux != self.device_main:
                labels = labels.cpu().to(dtype=torch.long, device=self.device_aux)
            output = output.view(-1, output.size(-1))
            loss = loss_fct(output, labels)

            if torch.isnan(loss) or torch.isinf(loss):
                print("NaN or Inf detected in loss!")
                exit()

            return loss
        return output

    def _sparsify(self, x):
        topk = torch.topk(x, k=self.k, dim=-1)
        values = self.sparse_act(topk.values + 1e-2)  # Prevent zero gradients
        result = torch.zeros_like(x)
        result.scatter_(-1, topk.indices, values)
        return result

    def clip_gradients(self):
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

    @staticmethod
    def _load_model(rank):
        print(f"----------Loading model ({model_id})----------")
        from transformers import AutoModelForCausalLM
        # from peft import LoraConfig
        # model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config, device_map=device, low_cpu_mem_usage=True,local_files_only=local_files_only)
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map=rank, return_dict=False,
                                                     return_dict_in_generate=True, output_hidden_states=True, low_cpu_mem_usage=True)
        model = model.half()
        return model
