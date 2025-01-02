import torch

# model_id = "facebook/opt-125m"
# model_id = "facebook/opt-350m"
# model_id = "facebook/opt-6.7b"
# model_id = "facebook/opt-2.7b"
# model_id = "meta-llama/Meta-Llama-3.1-8B"
# model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
# model_id = '/scratch/dg97/jm6832/Meta-Llama-3.1-8B/' # for local model
# model_id = "/scratch/dg97/jm6832/hf_home/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/48d6d0fc4e02fb1269b36940650a1b7233035cbb"
model_id = "openai-community/gpt2"
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
        self.k = sparsity_k
        enc_model = self._load_model(rank)
        self.encoder_0 = list(enc_model.children())[0].to(device=rank)
        # self.encoder_1 = list(enc_model.children())[1] # the final LM_head - not desirable here?
        self.embed_0 = torch.nn.Linear(in_features=768 * sequence_token_length, out_features=embed_dim).to(device=rank)
        self.embed_1 = torch.nn.Linear(in_features=embed_dim, out_features=768 * sequence_token_length).to(device=rank)
        dec_model = self._load_model(rank)
        decoder_embedding_layers_to_drop = 3
        self.decoder_0_list = \
            list(list(dec_model.children())[0].children())[decoder_embedding_layers_to_drop] \
            .to(device=rank)
        self.decoder_1 = \
            list(list(dec_model.children())[0].children())[4] \
            .to(device=rank)
        self.decoder_2 = list(dec_model.children())[1].to(device=rank)

    def forward(self, input_ids, labels=None, **kwargs):
        output = self.encoder_0(input_ids)[0]
        local_batch_size = input_ids.shape[0]
        sequence_length = input_ids.shape[1]
        output = output.view(local_batch_size, -1)
        output = self.embed_0(output)

        output = self._sparsify(output, self.k)

        output = self.embed_1(output)
        output = output.view(-1, sequence_length, 768)
        # output = torch.unsqueeze(output, 1).repeat(1, sequence_length, 1)
        for block in self.decoder_0_list:
            output = block(output, layer_past=None)[0]
        output = self.decoder_1(output)
        output = self.decoder_2(output)


        if labels is not None:

            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(output.view(-1, output.size(-1)), labels.view(-1))
            return loss
        return output

    @staticmethod
    def _sparsify(x, sparsity_k):
        ## Sparsity Function ref: https://github.com/openai/sparse_autoencoder/blob/main/sparse_autoencoder/model.py
        topk = torch.topk(x, k=sparsity_k, dim=-1)
        values = torch.nn.ReLU(topk.values)
        # make all other values 0
        result = torch.zeros_like(x)
        result.scatter_(-1, topk.indices, values)

    @staticmethod
    def _load_model(rank):
        print(f"----------Loading model ({model_id})----------")
        from transformers import AutoModelForCausalLM
        from peft import LoraConfig
        # model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config, device_map=device, low_cpu_mem_usage=True,local_files_only=local_files_only)
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map=rank, return_dict=False,
                                                     output_hidden_states=True, low_cpu_mem_usage=True)

        return model