#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import hydra
import lightning as L
import torch
import wandb
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from fashion.data.image_datamodule import ImageFashionDataModule
from fashion.models.visual_lightning import VisualFashionRecommender

PROJECT_ROOT = Path(__file__).parent.parent.parent
console = Console()


def check_device() -> str:
    if torch.backends.mps.is_available():
        console.print("[green]✓[/green] MPS (Metal Performance Shaders) available")
        return "mps"
    elif torch.cuda.is_available():
        console.print(f"[green]✓[/green] CUDA available: {torch.cuda.get_device_name(0)}")
        return "gpu"
    else:
        console.print("[yellow]![/yellow] No GPU available, using CPU")
        return "cpu"


def print_config(cfg: DictConfig) -> None:
    console.print(
        Panel.fit(
            OmegaConf.to_yaml(cfg),
            title="Configuration",
            border_style="blue",
        )
    )


def print_system_info() -> None:
    console.print("\n[bold blue]System Information[/bold blue]")
    console.print(f"  PyTorch version: {torch.__version__}")
    console.print(f"  Lightning version: {L.__version__}")
    console.print(f"  Python version: {sys.version.split()[0]}")

    if torch.backends.mps.is_available():
        console.print("  MPS backend: [green]Available[/green]")
    if torch.cuda.is_available():
        console.print(f"  CUDA version: {torch.version.cuda}")
    console.print()


@hydra.main(
    config_path=str(PROJECT_ROOT / "configs"), config_name="train_visual", version_base=None
)
def main(cfg: DictConfig) -> None:
    console.print(
        Panel.fit(
            "[bold]Visual Fashion Recommendation Training[/bold]\n"
            "ResNet18 + Attention-based History Aggregation",
            title="H&M Kaggle Competition",
            border_style="green",
        )
    )

    print_system_info()
    print_config(cfg)

    L.seed_everything(cfg.seed, workers=True)

    accelerator = check_device()
    if cfg.training.accelerator != "auto":
        accelerator = cfg.training.accelerator

    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        name=cfg.wandb.get("run_name"),
        save_dir=str(logs_dir),
        log_model=True,
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=["visual", "resnet18", "m1-mac"],
    )

    checkpoints_dir = PROJECT_ROOT / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        RichProgressBar(
            refresh_rate=1,
            leave=True,
        ),
        ModelCheckpoint(
            dirpath=checkpoints_dir,
            filename="visual-{epoch:02d}-{val/loss:.4f}-{val/map12:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=cfg.training.save_top_k,
            save_last=True,
            verbose=True,
        ),
        EarlyStopping(
            monitor="val/loss",
            patience=cfg.training.early_stopping_patience,
            mode="min",
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    console.print("\n[bold]Loading data...[/bold]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Initializing DataModule...", total=None)
        datamodule = ImageFashionDataModule(cfg.data)
        datamodule.setup(stage="fit")
        progress.update(task, description="[green]Data loaded![/green]")

    cfg.model.num_customers = datamodule.num_customers
    cfg.model.num_articles = datamodule.num_articles

    console.print("\n[bold]Dataset Statistics:[/bold]")
    console.print(f"  Customers: {datamodule.num_customers:,}")
    console.print(f"  Articles with images: {datamodule.num_articles:,}")
    console.print(f"  Training samples: {len(datamodule.train_dataset):,}")
    console.print(f"  Validation samples: {len(datamodule.val_dataset):,}")

    console.print("\n[bold]Initializing model...[/bold]")
    model = VisualFashionRecommender(cfg.model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"  Total parameters: {total_params:,}")
    console.print(f"  Trainable parameters: {trainable_params:,}")
    console.print(f"  Frozen parameters: {total_params - trainable_params:,}")

    trainer = L.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=accelerator,
        devices=cfg.training.devices,
        precision=cfg.training.precision,
        logger=wandb_logger,
        callbacks=callbacks,
        gradient_clip_val=cfg.training.gradient_clip_val,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        log_every_n_steps=cfg.training.log_every_n_steps,
        deterministic=cfg.training.get("deterministic", False),
        enable_progress_bar=True,
    )

    console.print("\n[bold green]Starting training...[/bold green]\n")
    trainer.fit(model, datamodule)

    console.print("\n[bold]Running test evaluation...[/bold]")
    trainer.test(model, datamodule)

    best_model_path = trainer.checkpoint_callback.best_model_path
    console.print(
        Panel.fit(
            f"[bold green]Training complete![/bold green]\n\n"
            f"Best checkpoint: {best_model_path}\n"
            f"W&B run: {wandb_logger.experiment.url}",
            title="Results",
            border_style="green",
        )
    )

    wandb.finish()


if __name__ == "__main__":
    main()
