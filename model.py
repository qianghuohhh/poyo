import os
import h5py
import torch
import collections
from typing import Tuple, Dict, List, Optional, Any, Mapping, Union
from poyo.data import Data,Dataset,collate
from poyo.models import POYO,POYOTokenizer
import torch_optimizer as optim
from torch.utils.data import DataLoader
from poyo.data.sampler import RandomFixedWindowSampler,SequentialFixedWindowSampler
import pytorch_lightning as pl

class POYOInterface(pl.LightningModule):
    r"""
        I know I'll end up regretting this name.
    """
    def __init__(
        self,
        *,
        dim=512,
        dim_head=64,
        num_latents=64,
        depth=2,
        cross_heads=1,
        self_heads=8,
        ffn_dropout=0.2,
        lin_dropout=0.4,
        atn_dropout=0.0,
        emb_init_scale=0.02,
        use_memory_efficient_attn=True,
    ):
        super().__init__() # store cfg
        self.save_hyperparameters(logger=False)
        self.model = POYO(
            dim=dim,
            dim_head=dim_head,
            num_latents=num_latents,
            depth=depth,
            cross_heads=cross_heads,
            self_heads=self_heads,
            ffn_dropout=ffn_dropout,
            lin_dropout=lin_dropout,
            atn_dropout=atn_dropout,
            emb_init_scale=emb_init_scale,
            use_memory_efficient_attn=use_memory_efficient_attn,
        )
        self.use_memory_efficient_attn=use_memory_efficient_attn

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.use_memory_efficient_attn:
            predict_value,loss,R2=self.model(
                        spike_unit_index=batch["spike_unit_index"],
                        spike_timestamps=batch["spike_timestamps"],
                        spike_type=batch["spike_type"],
                        #input_mask=data["input_mask"],
                        input_seqlen=batch["input_seqlen"],
                        # latent sequence
                        latent_index=batch["latent_index"],
                        latent_timestamps=batch["latent_timestamps"],
                        latent_seqlen=batch["latent_seqlen"],
                        # output sequence
                        session_index=batch["session_index"],
                        output_seqlen=batch["output_seqlen"],
                        output_timestamps=batch["output_timestamps"],
                        output_batch_index=batch["output_batch_index"],
                        output_values=batch["output_values"],
                        output_weights=batch["output_weights"]
                )
        else:
            predict_value,loss,R2=self.model(
                        spike_unit_index=batch["spike_unit_index"],
                        spike_timestamps=batch["spike_timestamps"],
                        spike_type=batch["spike_type"],
                        input_mask=batch["input_mask"],
                        # latent sequence
                        latent_index=batch["latent_index"],
                        latent_timestamps=batch["latent_timestamps"],
                        # output sequence
                        session_index=batch["session_index"],
                        output_timestamps=batch["output_timestamps"],
                        output_values=batch["output_values"],
                        output_weights=batch["output_weights"],
                        output_mask=batch["output_mask"]
                )
        output={}
        output["value"]=predict_value
        output["loss"]=loss
        output["R2"]=R2
        return output

    def configure_optimizers(self):
        optimizer = optim.Lamb(self.parameters(), lr=1e-4,weight_decay=1e-4)
        return optimizer

    def common_log(self, metrics, prefix='', **kwargs):
        for m in metrics:
            if m == "value":
                continue
            if m == "R2":
                labels = ['x', 'y', 'z']
                for i, r2 in enumerate(metrics[m]):
                    self.log(f'{prefix}_{m}_{labels[i]}', r2, **kwargs)
                self.log(f'{prefix}_{m}', metrics[m].mean(), **kwargs)
            else:
                self.log(f'{prefix}_{m}', metrics[m], **kwargs)
        

    def training_step(self, batch, batch_idx):
        output = self(batch)
        self.common_log(output, prefix='train')
        return output["loss"]

    def validation_step(self, batch, batch_idx):
        output = self(batch)
        self.common_log(output, prefix='valid')

    def test_step(self, batch, batch_idx):
        output = self(batch)
        self.common_log(output, prefix='test')
        return output