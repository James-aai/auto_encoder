
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from dataclasses import dataclass
import os
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
os.environ["CUDA_LAUNCH_BLOCKING"]="1"

model_id = "openai-community/gpt2"
# model_id = "facebook/opt-125m"
# model_id = "facebook/opt-350m"
# model_id = "facebook/opt-6.7b"
# model_id = "facebook/opt-2.7b"
# model_id = "meta-llama/Meta-Llama-3.1-8B"
# model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
# model_id = '/scratch/dg97/jm6832/Meta-Llama-3.1-8B/' # for local model
# model_id = "/scratch/dg97/jm6832/hf_home/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/48d6d0fc4e02fb1269b36940650a1b7233035cbb"
# local_files_only=True
local_files_only=False

#device = "cpu" #torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# device = torch.device('cuda') #if torch.cuda.is_available() else torch.device('cpu')
gpus = [ 0, 1 ]
# gpus = [0]
batch_size = 20 #* len(gpus)
learning_rate = 5e-5
row_limit = 250000 # data row size
epochs = 10
WORLD_SIZE = len(gpus) #torch.cuda.device_count()
embed_dim = 2

os.environ["HF_TOKEN"] = "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"

print(
    "GPUs Available: ", len(gpus)
    # [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
)
device="cuda:0"

@dataclass
class Config:
    wandb_project: str | None = "autoencoder-gpt2"
    wandb_name: str | None = "qsxim47qq-ppp"
    data_source_path: str = "/home/james/uni/phd/astro-ph-aic-cpt/data/*.parquet"
    processed_file = "abstracts_gpt2_tokens.pt"

cfg = Config()

def load_tokenizer():
    print("----------Loading tokenizer----------")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True,padding_side='right',local_files_only=local_files_only)
    # tokenizer.add_special_tokens({'pad_token': '<PAD>'})
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_model(rank):
    print(f"----------Loading model ({model_id})----------")
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig
    # model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config, device_map=device, low_cpu_mem_usage=True,local_files_only=local_files_only)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map=rank, return_dict=False, output_hidden_states=True, low_cpu_mem_usage=True)

    return model

# useful references:
# https://github.com/graykode/gpt-2-Pytorch/blob/master/GPT2/model.py
# https://stackoverflow.com/questions/73593136/typeerror-dropout-argument-input-position-1-must-be-tensor-not-tuple


class AutoEnc(torch.nn.Module):
    def __init__(self, rank):
        super(AutoEnc, self).__init__()
        enc_model = load_model(rank)
        self.encoder_0 = list(enc_model.children())[0].to(device=rank)
        # self.encoder_1 = list(enc_model.children())[1] # the final LM_head - not desirable here?
        self.embed_0 = torch.nn.Linear(in_features=768, out_features=embed_dim).to(device=rank)
        self.embed_1 = torch.nn.Linear(in_features=embed_dim, out_features=768).to(device=rank)
        dec_model = load_model(rank)
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
        # output = self.encoder_1(output)
        output = self.embed_0(output)
        # output = output.view([local_batch_size,-1])[:,0:embed_dim]
        embed_mask_ones = torch.ones(output.shape[1], embed_dim)
        embed_mask_zeros = torch.zeros(output.shape[1], output.shape[2] - embed_dim)
        embed_mask = torch.cat((embed_mask_ones, embed_mask_zeros), 0)
        output = output * embed_mask
        output = self.embed_1(output)
        print(torch.count_nonzero(output))
        # output = torch.topk()
        output = torch.unsqueeze(output, 1)
        for block in self.decoder_0_list:
            output = block(output, layer_past=None)[0]
        output = self.decoder_1(output)
        output = self.decoder_2(output)

        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(output.view(-1, output.size(-1)), labels.view(-1))
            return loss
        return output


def load_data():
    from pathlib import Path
    import glob
    my_file = Path(cfg.processed_file)
    max_sequence_length = 200
    # if my_file.is_file():
    #     print("--- loading pre-tokenized torch file... ---")
    #     input_ids = torch.load(processed_file)
    # else:
    ## dataloading is left as an exercise for the reader
    print("--- loading data from parquet files start... ---")
    filelist = glob.glob(cfg.data_source_path)
    df = pd.read_parquet(filelist).head(row_limit if row_limit is not None else -1)
    df['text'] = df['text'].astype('str')
    # tokenizer = AutoTokenizer.from_pretrained("gpt2")
    input_ids = []
    attention_mask = []

    # for row in df['text'].items():
    #     counter += 1
    #     if counter > row_limit: break
    #     tmp = tokenizer(row[1], return_tensors="pt")
    #     input_ids.append(tmp['input_ids'][0][:max_sequence_length])
    #     attention_mask.append(tmp['attention_mask'])

    # input_ids_nested = torch.nested.nested_tensor(input_ids, layout=torch.jagged)
    # input_ids_nested = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True)
    # torch.save(input_ids_nested, processed_file)
    df['text'] = df['text'].str.slice(0,max_sequence_length)

    data = [x[1] for x in df['text'].items()]
    return data


from torch.utils.data import Dataset
class TextDataset(Dataset):
    from torch import tensor
    def __init__(self, text):
        self.text = text

    def __getitem__(self, idx):
        # item = {key: self.tensor(val[idx]) for key, val in self.text.items()}
        item = {key: val[idx] for key, val in self.text.items()}
        # item['labels'] = self.tensor(self.text['input_ids'][idx])
        item['labels'] = self.text['input_ids'][idx]

        return item

    def __len__(self):
        return len(self.text['input_ids'])

def prep_data(text):
    print("----------Create train / test split and tokenize----------")
    from sklearn.model_selection import train_test_split
    prompts_train, prompts_test = train_test_split(prompts, test_size=.2)

    tokenizer = load_tokenizer()
    train_tokens = tokenizer(prompts_train, padding=True, return_tensors='pt')
    test_tokens = tokenizer(prompts_test, padding=True, return_tensors='pt')
    train_tokens = TextDataset(train_tokens)
    test_tokens = TextDataset(test_tokens)

    print("Done!")
    return train_tokens, test_tokens


def test_example(model, text):
    model.eval()
    tokenizer = load_tokenizer()
    if type(text) == 'str':
        encoded_input = tokenizer(text, return_tensors='pt').to(device)
        output = model(**encoded_input).to(device)
    else:
        encoded_input = text['input_ids'].to(device)
        encoded_input = torch.unsqueeze(encoded_input, 0)
        output = model(encoded_input).to(device)
        text = tokenizer.decode(text['input_ids'], skip_special_tokens=True)
    output_ids = [torch.argmax(x).item() for x in output.squeeze()]
    result = tokenizer.decode(output_ids, skip_special_tokens=True)

    print("Input:\n", text)
    print("output:\n", result)
    model.train()


from tqdm import tqdm
class TqdmExtraFormat(tqdm):
    """Provides a `total_time` format parameter"""

    @property
    def format_dict(self):
        d = super().format_dict
        total_time = d["elapsed"] * (d["total"] or 0) / max(d["n"], 1)
        d.update(total_time=self.format_interval(total_time) + " in total")
        return d

def progress_bar(iterable, **kwargs):
    return TqdmExtraFormat(**dict(
        iterable=iterable,
        unit="batch",
        mininterval=0,
        bar_format="{total_time}: {percentage:.0f}%|{bar}{r_bar}",
        **kwargs
    ))

# optim = Adafactor(model.parameters())
multi_gpu = False
def main(rank, train_dataset, test_dataset):
    # rank = gpus[rank]
    print("----training process started on GPU: ", rank)
    import torch
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from torch.utils.data.distributed import (
        DistributedSampler,
    )  # Distribute data across multiple gpus
    model = AutoEnc(rank)

    torch.cuda.set_device(rank)
    model.train()
    # model.device(rank)
    optim = AdamW(model.parameters(), lr=learning_rate)

    # from torcheval.metrics import WordErrorRate
    # metric = WordErrorRate(device=rank)

    if multi_gpu:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        torch.distributed.init_process_group("nccl", rank=rank, world_size=WORLD_SIZE)

        sampler_train = DistributedSampler(train_dataset)
        model = torch.nn.parallel.DistributedDataParallel(model, [rank])
    else:
        sampler_train = torch.utils.data.RandomSampler(train_dataset)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, sampler=sampler_train, num_workers=1
    )

    # model = torch.nn.DataParallel(model, gpus)
    # model = torch.nn.parallel.DistributedDataParallel(model, [rank])
    # model = torch.nn.DataParallel(model)

    for epoch in range(epochs):
        print("Starting epoch: ", epoch)
        if multi_gpu:
            sampler_train.set_epoch(epoch)
        with progress_bar(train_loader) as batches:
            batches.set_description(f"Epoch {epoch}")
            for batch in batches:
                input_ids = batch['input_ids'].to(rank)
                attention_mask = batch['attention_mask'].to(rank)
                labels = batch['labels'].to(rank)
                # forward pass
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

                if multi_gpu:
                    loss = outputs[0]
                else:
                    loss = outputs
                # print(loss.item())
                # backward pass
                optim.zero_grad()
                loss.sum().backward()
                # update weights
                optim.step()
                # print progress
                # metric.update(outputs[1], labels)
                # acc = (outputs.round() == labels).float().mean()
                batches.set_postfix(
                    loss=loss.sum()
                    # acc=float(acc)
                )
            # test_example(model.module, example_idx=2)
            test_example(model, text=test_dataset[0])


    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'loss': loss.sum(),
                }, "model.pt")

    # model.eval()
    # with torch.no_grad():
    #     ...
    #     out_data = model(data)

    # test_example(model.module, example_idx=6)
    # test_example(model.module, example_idx=10)
    if multi_gpu:
        torch.distributed.destroy_process_group()


# if __name__ == '__main__':
#     import torch.multiprocessing as mp
#     print("--- Main thread. World Size: ", WORLD_SIZE)
#     prompts = load_data()
#     train_dataset, test_dataset = prep_data(prompts)
#     mp.spawn(main, args=(train_dataset, test_dataset), nprocs=WORLD_SIZE)


prompts = load_data()
train_dataset, test_dataset = prep_data(prompts)
main(0, train_dataset, test_dataset)