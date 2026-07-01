# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import time

import hydra
import torch
import torch.nn as nn
import torch_geometric as pyg
import wandb

from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from torch_geometric.loader import DataLoader as PyGDataLoader

from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm  # ✅ ADDED

from physicsnemo.datapipes.gnn.hydrographnet_dataset import HydroGraphDataset
from physicsnemo.distributed.manager import DistributedManager
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.utils.logging.wandb import initialize_wandb
from physicsnemo.utils import load_checkpoint, save_checkpoint
from physicsnemo.models.meshgraphnet.meshgraphkan import MeshGraphKAN
from utils import compute_physics_loss


def collate_fn(batch):
    if isinstance(batch[0], tuple):
        graphs, physics_list = zip(*batch)
        batched_graph = pyg.data.from_data_list(graphs)
        physics_data = {}
        for key in physics_list[0].keys():
            physics_data[key] = torch.tensor(
                [d[key] for d in physics_list], dtype=torch.float
            )
        return batched_graph, physics_data
    else:
        return pyg.data.from_data_list(batch)


class MGNTrainer:
    def __init__(self, cfg: DictConfig, rank_zero_logger: RankZeroLoggingWrapper):
        assert DistributedManager.is_initialized()
        self.dist = DistributedManager()
        self.amp = cfg.amp
        self.noise_type = cfg.noise_type

        self.use_physics_loss = cfg.get("use_physics_loss", False)
        self.delta_t = cfg.get("delta_t", 1200.0)
        self.physics_loss_weight = cfg.get("physics_loss_weight", 1.0)

        mlp_act = "relu"
        if cfg.recompute_activation:
            mlp_act = "silu"

        dataset = HydroGraphDataset(
            name="hydrograph_dataset",
            data_dir=cfg.data_dir,
            prefix="M80",
            num_samples=500,
            n_time_steps=cfg.n_time_steps,
            k=4,
            noise_type=cfg.noise_type,
            noise_std=0.01,
            hydrograph_ids_file="train.txt",
            split="train",
            return_physics=self.use_physics_loss,
        )

        sampler = DistributedSampler(
            dataset,
            shuffle=True,
            drop_last=True,
            num_replicas=self.dist.world_size,
            rank=self.dist.rank,
        )

        self.dataloader = PyGDataLoader(
            dataset,
            batch_size=cfg.batch_size,
            sampler=sampler,
            pin_memory=True,
            num_workers=cfg.num_dataloader_workers,
            collate_fn=collate_fn,
        )

        self.model = MeshGraphKAN(
            cfg.num_input_features,
            cfg.num_edge_features,
            cfg.num_output_features,
            mlp_activation_fn=mlp_act,
            do_concat_trick=cfg.do_concat_trick,
            num_processor_checkpoint_segments=cfg.num_processor_checkpoint_segments,
            recompute_activation=cfg.recompute_activation,
        ).to(self.dist.device)

        if self.dist.world_size > 1:
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[self.dist.local_rank],
                output_device=self.dist.device,
                broadcast_buffers=self.dist.broadcast_buffers,
                find_unused_parameters=self.dist.find_unused_parameters,
            )

        self.model.train()
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda epoch: cfg.lr_decay_rate**epoch
        )

        self.scaler = GradScaler()

        self.epoch_init = load_checkpoint(
            to_absolute_path(cfg.ckpt_path),
            models=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            device=self.dist.device,
        )

    def train(self, batch):
        if self.use_physics_loss:
            graph, physics_data = batch
        else:
            graph = batch
            physics_data = None

        graph = graph.to(self.dist.device)
        
        # >>> ADDED: Strict NaN scrubbing for graph tensors to prevent poisoning from faulty standard deviations
        graph.x = torch.nan_to_num(graph.x, nan=0.0, posinf=0.0, neginf=0.0)
        graph.y = torch.nan_to_num(graph.y, nan=0.0, posinf=0.0, neginf=0.0)
        if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
            graph.edge_attr = torch.nan_to_num(graph.edge_attr, nan=0.0, posinf=0.0, neginf=0.0)
        
        if physics_data is not None:
            physics_data = {k: v.to(self.dist.device) for k, v in physics_data.items()}
            # >>> ADDED: Sanitize physics data tensors
            physics_data = {k: torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0) for k, v in physics_data.items()}

        self.optimizer.zero_grad()
        loss, loss_dict = self.forward(graph, physics_data)
        self.backward(loss)
        self.scheduler.step()
        return loss, loss_dict

    def forward(self, graph, physics_data):
        with autocast(device_type=self.dist.device.type, enabled=self.amp):
            pred = self.model(graph.x, graph.edge_attr, graph)
            
            # >>> ADDED: Clamp predictions to prevent FP16 numerical explosions during early unstable epochs
            pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
            pred = torch.clamp(pred, min=-10.0, max=10.0)
            
            mse_loss = self.criterion(pred, graph.y)
            loss = mse_loss
            loss_dict = {"total_loss": loss, "mse_loss": mse_loss}

            if self.use_physics_loss and physics_data is not None:
                phy_loss = compute_physics_loss(
                    pred, physics_data, graph, delta_t=self.delta_t
                )
                loss = loss + self.physics_loss_weight * phy_loss
                loss_dict["physics_loss"] = phy_loss

        return loss, loss_dict

    def backward(self, loss):
        if self.amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    DistributedManager.initialize()
    dist = DistributedManager()

    initialize_wandb(
        project="Modulus-Launch",
        entity="Modulus",
        name="Vortex_Shedding-Training",
        group="Vortex_Shedding-DDP-Group",
        mode=cfg.wandb_mode,
    )

    logger = PythonLogger("main")
    rank_zero_logger = RankZeroLoggingWrapper(logger, dist)
    rank_zero_logger.file_logging()

    trainer = MGNTrainer(cfg, rank_zero_logger)

    for epoch in tqdm(
        range(trainer.epoch_init, cfg.epochs),
        desc="Epochs",
        disable=dist.rank != 0,
    ):
        epoch_loss = 0.0
        num_batches = 0

        dataloader_tqdm = tqdm(
            trainer.dataloader,
            desc=f"Epoch {epoch}",
            leave=False,
            disable=dist.rank != 0,
        )

        for batch in dataloader_tqdm:
            loss, loss_dict = trainer.train(batch)

            batch_loss = loss.detach().item()
            epoch_loss += batch_loss
            num_batches += 1

            if torch.cuda.is_available():
                mem_alloc = torch.cuda.memory_allocated() / 1024**3
                mem_reserved = torch.cuda.memory_reserved() / 1024**3
            else:
                mem_alloc = 0.0
                mem_reserved = 0.0

            dataloader_tqdm.set_postfix(
                loss=f"{batch_loss:.4e}",
                gpu_mem=f"{mem_alloc:.2f}/{mem_reserved:.2f} GB",
            )

        avg_loss = epoch_loss / num_batches
        rank_zero_logger.info(f"Epoch {epoch} | Avg Loss: {avg_loss:.4e}")

        if dist.rank == 0:
            save_checkpoint(
                to_absolute_path(cfg.ckpt_path),
                models=trainer.model,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                scaler=trainer.scaler,
                epoch=epoch,
            )


if __name__ == "__main__":
    main()