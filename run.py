import os
import h5py
import torch
import collections
from poyo.models import POYO, POYOTokenizer
from poyo.data import Data,Dataset,collate
import torch_optimizer as optim
from torch.utils.data import DataLoader
from poyo.data.sampler import RandomFixedWindowSampler,SequentialFixedWindowSampler


if torch.cuda.is_available():
        device = torch.device("cuda")
else:
        device = torch.device("cpu")
torch.set_default_tensor_type(torch.DoubleTensor)
model=POYO(use_memory_efficient_attn=False).to(device)
tokenizer=POYOTokenizer(
        model.unit_tokenizer,
        model.session_tokenizer,
        latent_step=64,
        num_latents_per_step=64,
        using_memory_efficient_attn=False
        )
dataset=Dataset(
        root='/GPFS/yuezhifeng_lab_permanent/lutong/poyo/',
        split='train',
        include=[{
                "selection":[{
                        "dandiset":'processed/'
                }],
        }],
        transform=tokenizer,
)
sampler=RandomFixedWindowSampler(
        interval_dict=dataset.get_sampling_intervals(),
        window_length=5.0,
        generator=None,
)
dataloader=DataLoader(
        dataset=dataset,
        batch_size=32,
        sampler=sampler,
        collate_fn=collate
)
optimizer = optim.Lamb(model.parameters(), lr=1e-4)
for i in range(30):
        for data in dataloader:
                optimizer.zero_grad()
                '''loss=model(
                        spike_unit_index=data["spike_unit_index"].to(device),
                        spike_timestamps=data["spike_timestamps"].to(device),
                        spike_type=data["spike_type"].to(device),
                        #input_mask=data["input_mask"],
                        input_seqlen=data["input_seqlen"].to(device),
                        # latent sequence
                        latent_index=data["latent_index"].to(device),
                        latent_timestamps=data["latent_timestamps"].to(device),
                        latent_seqlen=data["latent_seqlen"].to(device),
                        # output sequence
                        session_index=data["session_index"].to(device),
                        output_seqlen=data["output_seqlen"].to(device),
                        output_timestamps=data["output_timestamps"].to(device),
                        output_batch_index=data["output_batch_index"].to(device),
                        output_values=data["output_values"].to(device),
                        output_weights=data["output_weights"].to(device)
                )'''
                output,loss=model(
                        spike_unit_index=data["spike_unit_index"].to(device),
                        spike_timestamps=data["spike_timestamps"].to(device),
                        spike_type=data["spike_type"].to(device),
                        input_mask=data["input_mask"].to(device),
                        # latent sequence
                        latent_index=data["latent_index"].to(device),
                        latent_timestamps=data["latent_timestamps"].to(device),
                        # output sequence
                        session_index=data["session_index"].to(device),
                        output_timestamps=data["output_timestamps"].to(device),
                        output_values=data["output_values"].to(device),
                        output_weights=data["output_weights"].to(device),
                        output_mask=data["output_mask"].to(device)
                )
                print(loss)
                loss.backward()
                optimizer.step()


    
