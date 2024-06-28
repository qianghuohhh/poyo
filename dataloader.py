import os
import h5py
import torch
import collections
from poyo.data import Data,Dataset,collate
from poyo.models import POYOTokenizer
from poyo.transforms import UnitDropout,Compose
import torch_optimizer as optim
from torch.utils.data import DataLoader
from poyo.data.sampler import RandomFixedWindowSampler,SequentialFixedWindowSampler
import pytorch_lightning as pl

class POYODataLoader(pl.LightningDataModule):
    def __init__(
        self,
        root,
        include,
        unit_tokenizer,
        session_tokenizer,
        latent_step,
        num_latents_per_step,
        batch_size=128,
        window_length=1.0,
        generator=None,
        sampler_random: bool = True,
        using_memory_efficient_attn: bool = True,
        ):
        super().__init__()
        self.root = root
        self.include = include
        self.unit_tokenizer = unit_tokenizer
        self.session_tokenizer = session_tokenizer
        self.latent_step = latent_step
        self.num_latents_per_step = num_latents_per_step
        self.batch_size=batch_size
        self.window_length = window_length
        self.generator = generator
        self.sampler_random = sampler_random
        self.using_memory_efficient_attn = using_memory_efficient_attn

    def prepare_data(self):
        pass

    def _generator_dataloader(self,split):
        tokenizer=POYOTokenizer(
            self.unit_tokenizer,
            self.session_tokenizer,
            latent_step=self.latent_step,
            num_latents_per_step=self.num_latents_per_step,
            using_memory_efficient_attn=self.using_memory_efficient_attn,
            eval=False if split=="train" else True
            )
        unit_dropout = UnitDropout()
        compose = Compose([unit_dropout,tokenizer])
        dataset=Dataset(
            root=self.root,
            split=split,
            include=self.include,
            transform=compose if split=="train" else tokenizer,
        )
        if split=="train":#self.sampler_random:
            sampler=RandomFixedWindowSampler(
                interval_dict=dataset.get_sampling_intervals(),
                window_length=self.window_length,
                generator=self.generator,
            )
        else:
            sampler=SequentialFixedWindowSampler(
                interval_dict=dataset.get_sampling_intervals(),
                window_length=self.window_length,
            )
        dataloader=DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            collate_fn=collate
        )
        return dataloader

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        # 实现数据集的定义，每张GPU都会执行该函数, stage 用于标记是用于什么阶段
        if stage == 'fit' or stage is None:
            self.train = self._generator_dataloader("train")
            self.valid = self._generator_dataloader("valid")
        if stage == 'test' or stage is None:
            self.test = self._generator_dataloader("test")
        

    # 以下方法创建不同阶段的数据加载器
    def train_dataloader(self):
        return self.train

    def val_dataloader(self):
        return self.valid

    def test_dataloader(self):
        return self.test

    def predict_dataloader(self):
        pass