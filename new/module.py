import torch
import transformers
from transformers import AutoModelForCausalLM
import pytorch_lightning as pl
import pytorch_lightning.utilities as U
from schedule import cosine_lr_lambda
from functools import partial
from printer import print0
import os, sys, time
import json

# from dataset import MyDataset

# 定义 LightningModule
class HFLM(pl.LightningModule):
    def __init__(self, model, dataset, args):
        super().__init__()
        self.args = args
        self.dataset = dataset

        def set_attr_dict(s, d):
            if not d: return
            d = json.loads(d)
            assert isinstance(d, dict)
            for k, v in d.items():
                if hasattr(s, k):
                    setattr(s, k, v)
                    print0(f"Model {s} attribute `{k}` set to `{v}`!")

        set_attr_dict(model, args.model_args)
        set_attr_dict(model.config, args.config_args)

        # if args.attn_implementation and hasattr(model, '_attn_implementation'):
        #     model._attn_implementation = args.attn_implementation
        with torch.no_grad():
            model.model.embeddings.weight[24].copy_(model.model.embeddings.weight[275].detach().clone())
            model.model.embeddings.weight[26].copy_(model.model.embeddings.weight[42].detach().clone())
            model.lm_head.weight[24].copy_(model.lm_head.weight[275].detach().clone())
            model.lm_head.weight[26].copy_(model.lm_head.weight[42].detach().clone())
        model.train()
        if args.compile_model:
            self.model = torch.compile(model)
        else:
            self.model = model

        # Initialize variables for throughput calculation and EMA smoothing
        self.ema_throughput = 0.0  # Exponential Moving Average of throughput
        self.last_time = time.time()  # Timestamp for measuring time per step

    def forward(self, x, y):
        # Shift labels: requires transformers 4.50.3 and above.
        return self.model(input_ids=x, labels="Anything except None", shift_labels=y)

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x, y)
        loss = outputs.loss
        self.log("loss", loss, prog_bar=True, logger=True)
        self.log("lr", self.lr_schedulers().get_lr()[0])
        return loss

    def on_train_epoch_start(self):
        self.dataset.global_rank = self.global_rank
        self.dataset.real_epoch = self.current_epoch
        self.dataset.world_size = self.trainer.world_size
    
    def on_before_optimizer_step(self, optimizer):
        grad_norms = U.grad_norm(self, norm_type=2.0)
        self.log('grad_2.0_norm_total', grad_norms['grad_2.0_norm_total'])

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Compute percentage of training completed
        percent_completed = self.global_step / self.args.total_steps
        self.log("percent", percent_completed)

        # Measure time per step
        current_time = time.time()
        time_this_step = current_time - self.last_time
        self.last_time = current_time 

        # Compute raw throughput (tokens per second)
        tokens_per_step = self.args.real_bsz * self.args.ctx_len
        raw_throughput = tokens_per_step / time_this_step

        self.ema_throughput = 0.9 * self.ema_throughput + 0.1 * raw_throughput
        self.log("throughput", self.ema_throughput, prog_bar=True, logger=True)

        if self.ema_throughput > 0 and self.trainer.global_step >= 40:
            remaining_steps = self.args.total_steps - self.global_step
            remaining_tokens = remaining_steps * tokens_per_step
            eta_seconds = remaining_tokens / self.ema_throughput
            eta_minutes = eta_seconds / 60  # Convert seconds to minutes
            self.log("eta", eta_minutes, prog_bar=True, logger=True)
        
        if self.global_step >= self.args.total_steps:
            self.model.save_pretrained(
                os.path.join(self.args.save_path, f"steps_{self.args.total_steps}_final")
            )
            sys.exit(0)

    def configure_optimizers(self):
        def is_matrix(shape):
            prod = 1
            for d in shape:
                if d >= 8:
                    prod *= d
            return prod >= 16384

        optim_groups = []
        # 这里需要对于1D和0D的单列出来
        if self.args.weight_decay > 0 and self.args.no_decay_1d:
            decay_set = []
            nodecay_set = []
            another_set = []
            for n, p in self.model.named_parameters():
                print0(f"{n:60}", end='')
                add_decay = is_matrix(p.shape)
                print0(f"{int(add_decay)}   ", list(p.shape))
                if self.args.rwkv_lr and '.w0' in n:
                    print(f"RWKV 2x: {n}")
                    another_set.append(p)
                elif add_decay:
                    decay_set.append(p)
                else:
                    nodecay_set.append(p)
            if self.args.rwkv_lr:
                optim_groups = [
                    {"params": decay_set, "weight_decay": self.args.weight_decay, "lr": self.args.lr},
                    {"params": nodecay_set, "weight_decay": 0.0, "lr": self.args.lr},
                    {"params": another_set, "weight_decay": 0.0, "lr": 2*self.args.lr},
                ]
            else:
                optim_groups = [
                    {"params": decay_set, "weight_decay": self.args.weight_decay},
                    {"params": nodecay_set, "weight_decay": 0.0},
                ]

        # if 'deepspeed' in self.args.strategy:
        from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
        optimizer = (DeepSpeedCPUAdam if 'offload' in self.args.strategy else FusedAdam)(
            optim_groups if optim_groups else self.model.parameters(),
            lr=self.args.lr,
            betas=(self.args.beta1, self.args.beta2),
            weight_decay=self.args.weight_decay,
            eps=self.args.epsilon,
        )

        # 使用 LambdaLR 实现自定义调度器
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, 
            lr_lambda=partial(
                cosine_lr_lambda, 
                total_steps=self.args.total_steps, warmup_steps=self.args.warmup_steps,
                lr=self.args.lr, lr_end=self.args.lr_end,
            )
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # 按 step 更新学习率
                "frequency": 1       # 每个 step 都更新
            }
        }
    
    def train_dataloader(self):
        self.dataset.global_rank = self.global_rank
        self.dataset.real_epoch = self.current_epoch
        self.dataset.world_size = self.trainer.world_size
        return torch.utils.data.DataLoader(
            self.dataset,
            shuffle=False,
            pin_memory=('gpu' in self.args.accelerator.lower()),
            batch_size=self.args.bsz,
            num_workers=1,
            persistent_workers=False,
            drop_last=True
        )