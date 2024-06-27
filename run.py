import os
import h5py
import torch
from pathlib import Path
import collections
from poyo.models import POYO, POYOTokenizer
from poyo.data import Data,Dataset,collate
from dataloader import POYODataLoader
import torch_optimizer as optim

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor
)

from model import POYOInterface
from poyo.data.sampler import RandomFixedWindowSampler,SequentialFixedWindowSampler

if torch.cuda.is_available():
        device = torch.device("cuda")
else:
        device = torch.device("cpu")
is_distributed = torch.cuda.device_count() > 1
nodes=1 #torch.cuda.device_count()
default_strat = 'auto' if pl.__version__.startswith('2.0') else 'auto'
""" torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True) """

os.environ["WANDB_API_KEY"] = 'eb9f8bffb20b4fbb7550ae857caad45673e6ebb8'
pl.seed_everything(seed=0)
wandb_project="poyo"
epochs=400
max_steps=1000000000
log_every_n_steps=1
half_precision=True
use_memory_efficient_attn=False
default_root_dir=Path("./data/runs").resolve()
gradient_clip_val=1.0
#accumulate_batches=2
profiler=""
overfit_batches=False

wandb_logger = WandbLogger(
        project=wandb_project,
        save_dir=default_root_dir,
    )

model=POYOInterface(
        epochs=epochs,
        dim=128,
        dim_head=64,
        num_latents=128,
        depth=6,
        cross_heads=8,
        self_heads=8,
        ffn_dropout=0.3,
        lin_dropout=0.3,
        atn_dropout=0.3,
        emb_init_scale=0.02,
        use_memory_efficient_attn=use_memory_efficient_attn,
).to(device)

data_module = POYODataLoader(
        root = '/GPFS/yuezhifeng_lab_permanent/lutong/poyo/',
        include = [{
                "selection":[{
                        "dandiset":'processed/',
                        "session": "c_20161013_center_out_reaching"
                }],
        }],
        unit_tokenizer = model.model.unit_tokenizer,
        session_tokenizer = model.model.session_tokenizer,
        latent_step = 0.1,
        num_latents_per_step = 64,
        batch_size = 128 * torch.cuda.device_count(),
        using_memory_efficient_attn=use_memory_efficient_attn,
)

callbacks=[]
lr_monitor = LearningRateMonitor(logging_interval='step')
callbacks.append(lr_monitor)

trainer = pl.Trainer(
        logger=wandb_logger,
        max_epochs=epochs,
        max_steps=max_steps,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=torch.cuda.device_count() if torch.cuda.is_available() else None,
        num_nodes=nodes,
        check_val_every_n_epoch=1,
        log_every_n_steps=log_every_n_steps,
        # val_check_interval=cfg.train.val_check_interval,
        callbacks=callbacks,
        default_root_dir=default_root_dir,
        # track_grad_norm=2 if cfg.train.log_grad else -1, # this is quite cluttered, but probably better that way. See https://github.com/Lightning-AI/lightning/issues/1462#issuecomment-1190253742 for patch if needed, though.
        precision=16 if half_precision else 32,
        strategy=DDPStrategy(find_unused_parameters=True) if is_distributed else default_strat,
        gradient_clip_val=gradient_clip_val,
        #accumulate_grad_batches=accumulate_batches,
        #profiler=profiler if profiler else None,
        #overfit_batches=1 if overfit_batches else 0
    ) 

trainer.fit(model, datamodule=data_module)
trainer.test(model, datamodule=data_module)

    
