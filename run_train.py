
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from dataclasses import dataclass
import os, sys, socket
import pandas as pd
# from models import gpt2_autoenc as model_def ## non-sparse
# from models import llama3_1_SAE as model_def
# from models import gpt2_SAE as model_def
from models import gpt2_SAE_v2 as model_def
# from models import bart_SAE as model_def

from torch.optim import AdamW as optimiser_class
# from torch.optim import SGD as optimiser_class

from contextlib import redirect_stdout
import wandb
import pathlib
from contextlib import nullcontext, redirect_stdout

os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["CUDA_LAUNCH_BLOCKING"]="1"
os.environ["TOKENIZERS_PARALLELISM"]="true"
os.environ["HF_TOKEN"] = "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"
os.environ["WANDB_API_KEY"] = "5df70ce8ab7fcc1725fa5e5865f8eeed71514b09"

model_name = model_def.model_id.split('/')[-1]
gpus = [ 0 ]
multi_gpu = (len(gpus) > 1)
# batch_size = 1 * len(gpus)
learning_rate = 5e-5 * 10
row_limit = 1000 # data row size. null = all data. 260,000 is most of it
epochs = 10
steps_per_update = 1
default_batch_size = 10
epoch_start = 0
WORLD_SIZE = len(gpus) #torch.cuda.device_count()
embed_dim = 1000
sparsity_k = 100
sequence_token_length = 300
load_checkpoint = False
print_output_to_file = True
print_output_file_path = "./print_output/"
use_profiler = False
save_checkpoints = False
checkpoint_path = "./checkpoints/"
experiment_name = "sparse-autoencoder-" + model_name + "-" + str(embed_dim) + "-" + str(sparsity_k)
checkpoint_prefix = f"{checkpoint_path}{experiment_name}"

if not os.path.exists(print_output_file_path):
    os.makedirs(print_output_file_path)

if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

with open(print_output_file_path + "init.txt", "a+") if print_output_to_file else nullcontext() as f:
    with redirect_stdout(f) if print_output_to_file else nullcontext():
        print( "GPUs Available: ", len(gpus),
            [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        )

@dataclass
class Config:
    wandb_project: str | None = experiment_name
    wandb_name: str | None = "qsxim47qq-ppp"
    data_source_path: str = ("/home/659/jm6832/data/astro-ph-aic-cpt/data" if "gadi" in socket.gethostname() else (
            ("d:/@" if os.name == "nt" else "/") + "home/james/uni/phd/astro-ph-aic-cpt/data/*.parquet"))
    processed_file: str = "abstracts_tokens_" + model_name + ".pt"

cfg = Config()

def load_data():
    # print(f"----------Loading data from disc ----------")
    from pathlib import Path
    import glob
    my_file = Path(cfg.processed_file)
    # if my_file.is_file():
    #     print("--- loading pre-tokenized torch file... ---")
    #     input_ids = torch.load(processed_file)
    # else:
    ## dataloading is left as an exercise for the reader
    with open(print_output_file_path + "init.txt", "a+") if print_output_to_file else nullcontext() as f:
        with redirect_stdout(f) if print_output_to_file else nullcontext():
            print("--- loading data from parquet files start... ---")
    filelist = glob.glob(cfg.data_source_path)
    assert len(filelist) > 0
    filelist = [x.replace("\\", "/") for x in filelist]
    df = pd.read_parquet(filelist).head(row_limit if row_limit is not None else -1)
    df['text'] = df['text'].astype('str')
    df['text'] = df['text'].str.slice(0,sequence_token_length * 5)

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
    with open(print_output_file_path + "init.txt", "a+") if print_output_to_file else nullcontext() as f:
        with redirect_stdout(f) if print_output_to_file else nullcontext():
            print("----------Create train / test split and tokenize----------")
    from sklearn.model_selection import train_test_split
    prompts_train, prompts_test = train_test_split(text, test_size=.2)

    tokenizer = model_def.load_tokenizer()
    train_tokens = tokenizer(prompts_train, max_length=sequence_token_length, truncation=True
                             , pad_to_multiple_of=sequence_token_length, padding="max_length", return_tensors='pt')
    test_tokens = tokenizer(prompts_test, max_length=sequence_token_length, truncation=True
                            , pad_to_multiple_of=sequence_token_length, padding="max_length", return_tensors='pt')
    train_tokens = TextDataset(train_tokens)
    test_tokens = TextDataset(test_tokens)

    print("Done!")
    return train_tokens, test_tokens

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n) and p.grad is not None:
            layers.append(n)
            ave_grad = p.grad.abs().mean().cpu()
            max_grad = p.grad.abs().max().cpu()
            print(f"Layer {n} with average grad {ave_grad:.5f} and max_grad {max_grad:.5f}")
            ave_grads.append(ave_grad)
            max_grads.append(max_grad)
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.show()

def test_example(model, text):
    model.eval()
    tokenizer = model_def.load_tokenizer()
    if type(text) == 'str':
        encoded_input = tokenizer(text, return_tensors='pt').to(gpus[0])
        output = model(**encoded_input).to(gpus[0])
    else:
        encoded_input = text['input_ids'].to(gpus[0])
        encoded_input = torch.unsqueeze(encoded_input, 0)
        output = model(encoded_input).to(gpus[0])
        text = tokenizer.decode(text['input_ids'], skip_special_tokens=True)
    output_ids = [torch.argmax(x).item() for x in output.squeeze()]
    result = tokenizer.decode(output_ids, skip_special_tokens=True)

    model.train()
    return result, text

def test_example_print(model, text_ids, epoch):
    result, text_str = test_example(model, text_ids)
    with open(print_output_file_path + f"e{epoch}.txt", "a+") if print_output_to_file else nullcontext() as f:
        with redirect_stdout(f) if print_output_to_file else nullcontext():
            print(f"Input: {text_str}")
            print(f"Result: {"".join(result.splitlines())}")


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



def main(rank, batch_size, train_dataset, test_dataset):
    # rank = gpus[rank]
    print(f"----training process started for model {model_name} on GPU: {rank}")
    import torch
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from torch.profiler import profile, record_function, ProfilerActivity


    class Logger:
        def __init__(self, **kws):
            self.vals = {}
            self.enabled = (rank == 0) and not kws.pop("dummy", False)
            if self.enabled:
                wandb.init(
                    **kws
                )
    
        def logkv(self, k, v):
            if self.enabled:
                self.vals[k] = v.detach() if isinstance(v, torch.Tensor) else v
            return v
    
        def dumpkvs(self):
            if self.enabled:
                wandb.log(self.vals)
                self.vals = {}

    #if logger is None:
    logger = Logger(project=f"sparse-autoenc-{model_name}-emb-{embed_dim}-k-{sparsity_k}")

    with (torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(F'./log/{model_name}_sparseautoenc'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
    ) if use_profiler else nullcontext() as prof):

        model = model_def.SparseAutoEnc(rank, sequence_token_length, embed_dim, sparsity_k)
        optim = optimiser_class(model.parameters(), lr=learning_rate)
        update_step_counter = 0
        epoch_start = 0

        if load_checkpoint:
            ## find the file matching checkpoint prefix with the highest epoch number
            # os.find(checkpoint_prefix)
            checkpoint = torch.load(f"./checkpoints/SparseAutoEnc_{model_name}_tokens_{sequence_token_length}.pt", weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            optim.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch_start = checkpoint['epoch']
            loss = checkpoint['loss']

        torch.cuda.set_device(rank)
        model.train()
        optim.zero_grad()

        # validation metric
        from torcheval.metrics import WordErrorRate
        val_metric = WordErrorRate(device=rank)

        if multi_gpu:
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            torch.distributed.init_process_group(
                backend = "gloo" if os.name == "nt" or not torch.cuda.is_available() else "nccl",
                rank=rank, world_size=WORLD_SIZE,
                init_method = "env://?use_libuv=False" if os.name == "nt" else "env://")

            sampler_train = DistributedSampler(train_dataset)
            model = torch.nn.parallel.DistributedDataParallel(model, [rank])
        else:
            sampler_train = torch.utils.data.RandomSampler(train_dataset)

        train_loader = DataLoader(
            train_dataset, batch_size=(batch_size * WORLD_SIZE), shuffle=False, sampler=sampler_train, num_workers=1
        )
        # only needed for memory debugging - turn off for regular use
        # torch.cuda.memory._record_memory_history()

        for epoch in range(epoch_start, epoch_start + epochs):

            with open(print_output_file_path + f"e{epoch}.txt", "a+") if rank == 0 and print_output_to_file else nullcontext() as f:
                with redirect_stdout(f) if rank == 0 and print_output_to_file else nullcontext():
                    if rank == 0:
                        print("Starting epoch: ", epoch)
                    if multi_gpu:
                        sampler_train.set_epoch(epoch)
                    # with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:

                    with progress_bar(train_loader) as batches:
                        batches.set_description(f"Epoch {epoch}")
                        for batch in batches:
                            if use_profiler:
                                prof.step()
                            # with torch.amp.autocast(device_type="cuda"):
                            with record_function("data_loading") if use_profiler else nullcontext():
                                input_ids = batch['input_ids'].to(rank)
                                attention_mask = batch['attention_mask'].to(rank)
                                labels = batch['labels'].to(rank)
                            # forward pass
                            with record_function("forward_pass") if use_profiler else nullcontext():
                                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

                            # torch.cuda.memory._dump_snapshot("my_snapshot.pickle")

                            with record_function("backward_pass") if use_profiler else nullcontext():
                                loss = outputs #.cpu()
                                update_step_counter += batch_size
                                if update_step_counter >= steps_per_update:
                                    update_step_counter = 0
                                    optim.zero_grad()
                                    loss.backward()
                                    # update weights
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                                    optim.step()
                                    # optim.zero_grad()
                                else:
                                    loss.backward()
                                    # update weights
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                                    optim.step()
                                wandb.log({"train_loss": loss.item()})

                            with record_function("update_info") if use_profiler else nullcontext():
                                # print progress
                                # metric.update(outputs[1], labels)
                                # acc = (outputs.round() == labels).float().mean()
                                batches.set_postfix(
                                    loss=loss.sum()
                                    # acc=float(acc)
                                )
                    logger.dumpkvs()

                    # Epoch finish up steps
                    if rank==0:
                        plot_grad_flow(model.named_parameters())
                        test_example_print(model, text_ids=test_dataset[0], epoch=epoch)
                        test_example_print(model, text_ids=test_dataset[1], epoch=epoch)
                        test_example_print(model, text_ids=test_dataset[2], epoch=epoch)
                        test_example_print(model, text_ids=test_dataset[3], epoch=epoch)
                        for i in range(0,min(len(test_dataset), 30)):
                            result, target = test_example(model, test_dataset[i])
                            val_metric.update(input=result, target=target)

                        with open(print_output_file_path + f"e{epoch}.txt", "a+") if print_output_to_file else nullcontext():
                            print(f"Word Error rate = {val_metric.compute()}")

                        # print(prof.key_averages(group_by_input_shape=True).table()) #sort_by="cpu_time_total", row_limit=10))
                        # prof.export_chrome_trace("~/chrom.trace")
                        # print(prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=10))

                        if save_checkpoints:
                            with open(print_output_file_path + f"e{epoch}.txt", "a+") if print_output_to_file else nullcontext():
                                print("Saving model... ")
                            torch.save({
                                        'epoch': epochs,
                                        'model_state_dict': model.state_dict(),
                                        'optimizer_state_dict': optim.state_dict(),
                                        'loss': loss.sum(),
                                        }, f"{checkpoint_prefix}_epochs_{epoch}.pt")
                            try:
                                pathlib.Path.unlink(f"{checkpoint_prefix}_epochs_{epoch-1}.pt")
                            except OSError as e:
                                continue

    if multi_gpu:
        torch.distributed.destroy_process_group()



if __name__ == '__main__':
    import torch.multiprocessing as mp
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', dest='batch_size', type=int, help='specify batch size')
    args = parser.parse_args()
    batch_size = args.batch_size
    if batch_size is None:
        batch_size = default_batch_size
    with open(print_output_file_path + "init.txt", "a+") if print_output_to_file else nullcontext():
        print ("--- Using batch size: ", batch_size)
        print("--- Main thread. World Size: ", WORLD_SIZE)



    prompts = load_data()
    train_dataset, test_dataset = prep_data(prompts)
    if WORLD_SIZE > 1:
        mp.spawn(main, args=(batch_size, train_dataset, test_dataset), nprocs=WORLD_SIZE) #, join=True)
    else:
        main(0, batch_size, train_dataset, test_dataset)


# prompts = load_data()
# train_dataset, test_dataset = prep_data(prompts)
# main(0, train_dataset, test_dataset)
