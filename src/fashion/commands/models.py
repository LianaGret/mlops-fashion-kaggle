from __future__ import annotations

import datetime
import subprocess
from pathlib import Path
from typing import Literal

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from fashion.console import (
    console,
    create_table,
    print_error,
    print_header,
    print_info,
    print_key_value,
    print_success,
    print_warning,
)

ModelType = Literal["baseline", "collaborative", "content", "hybrid", "visual"]

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"


def _find_best_checkpoint(model_type: str) -> Path | None:
    if not CHECKPOINTS_DIR.exists():
        return None

    is_visual = model_type == "visual"

    if is_visual:
        checkpoints = [p for p in CHECKPOINTS_DIR.glob("*.ckpt") if "visual" in p.name]
    else:
        checkpoints = [
            p
            for p in CHECKPOINTS_DIR.glob("*.ckpt")
            if "visual" not in p.name and p.name != "last.ckpt"
        ]

    if not checkpoints:
        return None

    return sorted(checkpoints, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def _run_command(cmd: list[str]) -> str | None:
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def _get_version_info() -> dict[str, str]:
    info = {}

    git_commit = _run_command(["git", "rev-parse", "HEAD"])
    if git_commit:
        info["git_commit"] = git_commit

    jj_change = _run_command(["jj", "log", "-r", "@", "--no-graph", "-T", "change_id"])
    if jj_change:
        info["jj_change_id"] = jj_change

    jj_commit = _run_command(["jj", "log", "-r", "@", "--no-graph", "-T", "commit_id"])
    if jj_commit:
        info["jj_commit_id"] = jj_commit

    return info


def _run_visual_inference(
        model,
    device,
    output_path: Path,
    batch_size: int,
    sample_customers: int,
) -> None:
    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image
    from torchvision import transforms
    from tqdm import tqdm

    data_dir = PROJECT_ROOT / "data" / "raw"

    print_info("Loading customer and article data...")
    transactions = pd.read_csv(data_dir / "transactions_train.csv")
    articles = pd.read_csv(data_dir / "articles.csv")

    images_dir = data_dir / "images"

    def get_image_path(article_id):
        article_str = str(article_id).zfill(10)
        return images_dir / article_str[:3] / f"{article_str}.jpg"

    valid_articles = [
        aid for aid in articles["article_id"].unique() if get_image_path(aid).exists()
    ]
    print_key_value("Articles with images", f"{len(valid_articles):,}")

    article_to_idx = {aid: idx for idx, aid in enumerate(valid_articles)}
    idx_to_article = {idx: aid for aid, idx in article_to_idx.items()}

    unique_customers = transactions["customer_id"].unique()
    sampled_customers = np.random.choice(
        unique_customers, size=min(sample_customers, len(unique_customers)), replace=False
    )
    print_key_value("Customers to score", f"{len(sampled_customers):,}")

    customer_histories = {}
    for cid, group in transactions.groupby("customer_id"):
        if cid in sampled_customers:
            history = [
                article_to_idx[aid] for aid in group["article_id"] if aid in article_to_idx
            ][-5:]
            if history:
                customer_histories[cid] = history

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def load_image(article_idx):
        article_id = idx_to_article[article_idx]
        path = get_image_path(article_id)
        try:
            img = Image.open(path).convert("RGB")
            return transform(img)
        except Exception:
            return torch.zeros(3, 224, 224)

    print_info("Encoding article images...")
    article_embeddings = []

    all_indices = list(range(len(valid_articles)))
    for i in tqdm(range(0, len(all_indices), batch_size), desc="Encoding articles"):
        batch_indices = all_indices[i : i + batch_size]
        batch_images = torch.stack([load_image(idx) for idx in batch_indices])
        batch_images = batch_images.to(device)

        with torch.no_grad():
            embeddings = model.model.encode_images(batch_images)
            article_embeddings.append(embeddings.cpu())

    article_embeddings = torch.cat(article_embeddings, dim=0).to(device)
    print_key_value("Article embeddings shape", str(article_embeddings.shape))

    print_info("Generating recommendations...")
    results = []

    for cid in tqdm(sampled_customers, desc="Scoring customers"):
        if cid not in customer_histories:
            top_articles = valid_articles[:12]
        else:
            history = customer_histories[cid]

            history_images = torch.stack([load_image(idx) for idx in history])
            while len(history_images) < 5:
                history_images = torch.cat([history_images, torch.zeros(1, 3, 224, 224)], dim=0)
            history_images = history_images[:5].unsqueeze(0).to(device)

            history_mask = torch.tensor(
                [[1.0] * min(len(history), 5) + [0.0] * (5 - min(len(history), 5))],
                device=device,
            )

            customer_idx = torch.tensor([0], device=device)

            with torch.no_grad():
                scores = model.model.score_articles(
                    customer_idx, history_images, history_mask, article_embeddings
                )

            top_indices = scores[0].argsort(descending=True)[:12].cpu().numpy()
            top_articles = [idx_to_article[idx] for idx in top_indices]

        prediction_str = " ".join(str(aid).zfill(10) for aid in top_articles)
        results.append({"customer_id": cid, "prediction": prediction_str})

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)

    print_success(f"Predictions saved to {output_path}")
    print_key_value("Total customers", str(len(results)))
    print_key_value("Format", "customer_id,prediction (12 article_ids)")


def _run_training(cfg: OmegaConf, resume: Path | None = None) -> None:
    import lightning as L
    import wandb
    from lightning.pytorch.callbacks import (
        EarlyStopping,
        LearningRateMonitor,
        ModelCheckpoint,
        RichProgressBar,
    )
    from lightning.pytorch.loggers import WandbLogger

    model_type = cfg.model.get("type", "hybrid")
    is_visual = model_type == "visual"

    if is_visual:
        from fashion.data.image_datamodule import ImageFashionDataModule
        from fashion.models.visual_lightning import VisualFashionRecommender
    else:
        from fashion.data.datamodule import FashionDataModule
        from fashion.models.lightning import FashionRecommender

    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    tags = list(cfg.wandb.get("tags", ["fashion", "recommendations"]))
    if is_visual:
        tags.extend(["visual", "resnet18"])

    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        name=cfg.wandb.run_name,
        entity=cfg.wandb.get("entity"),
        save_dir=str(logs_dir),
        log_model=True,
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=tags,
    )

    version_info = _get_version_info()
    if version_info:
        wandb_logger.experiment.config.update(version_info)

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    if is_visual:
        ckpt_filename = "visual-{epoch:02d}-{val/loss:.4f}"
        monitor_metric = "val/loss"
    else:
        ckpt_filename = "{epoch}-{val_loss:.4f}"
        monitor_metric = "val_loss"

    callbacks = [
        RichProgressBar(),
        ModelCheckpoint(
            dirpath=CHECKPOINTS_DIR,
            filename=ckpt_filename,
            monitor=monitor_metric,
            mode="min",
            save_top_k=cfg.training.save_top_k,
            save_last=True,
        ),
        EarlyStopping(
            monitor=monitor_metric,
            patience=cfg.training.early_stopping_patience,
            mode="min",
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    accelerator = cfg.training.accelerator
    if accelerator == "auto" and is_visual:
        import torch

        if torch.backends.mps.is_available():
            accelerator = "mps"
            print_info("Using MPS (Metal Performance Shaders) for training")

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
    )

    if is_visual:
        datamodule = ImageFashionDataModule(cfg.data)
        print_info("Setting up image dataset...")
        datamodule.setup(stage="fit")

        cfg.model.num_customers = datamodule.num_customers
        cfg.model.num_articles = datamodule.num_articles

        print_key_value("Customers", f"{datamodule.num_customers:,}")
        print_key_value("Articles with images", f"{datamodule.num_articles:,}")
        print_key_value("Training samples", f"{len(datamodule.train_dataset):,}")

        model = VisualFashionRecommender(cfg.model)
    else:
        datamodule = FashionDataModule(cfg.data)
        model = FashionRecommender(cfg.model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_key_value("Total parameters", f"{total_params:,}")
    print_key_value("Trainable parameters", f"{trainable_params:,}")
    console.print()

    resume_path = str(resume) if resume else None
    trainer.fit(model, datamodule, ckpt_path=resume_path)

    wandb.finish()
    print_success("Training complete!")
    print_key_value("Best checkpoint", str(trainer.checkpoint_callback.best_model_path))


class Models:
    """Model management commands.

    Commands:
        train       Train a model with specified configuration
        infer       Run inference with a trained model
        list        List available models and checkpoints
    """

    def train(
        self,
        config: str = "train",
        overrides: list[str] | None = None,
        resume: Path | None = None,
        dry_run: bool = False,
    ) -> None:
        """Train a model using PyTorch Lightning.

        Args:
            config: Hydra config name (default: train)
            overrides: Config overrides in key=value format
            resume: Path to checkpoint to resume from
            dry_run: Print config without running training
        """
        print_header("Model Training", "PyTorch Lightning + W&B")
        console.print()

        if not CONFIG_DIR.exists():
            print_error(f"Config directory not found: {CONFIG_DIR}")
            return

        try:
            with initialize_config_dir(config_dir=str(CONFIG_DIR.absolute()), version_base=None):
                cfg = compose(config_name=config, overrides=overrides or [])
        except Exception as error:
            print_error(f"Failed to load config: {error}")
            return

        print_info("Configuration loaded:")
        console.print()
        console.print(OmegaConf.to_yaml(cfg))

        if dry_run:
            print_warning("Dry run mode - not starting training")
            return

        if resume:
            print_info(f"Resuming from checkpoint: {resume}")

        print_info("Starting training...")
        console.print()

        _run_training(cfg, resume)

    def infer(
        self,
        checkpoint: Path | None = None,
        model_type: ModelType = "visual",
        input_path: Path | None = None,
        output_path: Path = Path("predictions.csv"),
        batch_size: int = 16,
        sample_customers: int = 100,
    ) -> None:
        """Run inference with a trained model.

        Args:
            checkpoint: Path to model checkpoint (uses best if not specified)
            model_type: Type of model to use (visual, hybrid, collaborative, content)
            input_path: Path to input data
            output_path: Path to save predictions
            batch_size: Batch size for inference
            sample_customers: Number of customers to generate predictions for
        """
        import pandas as pd
        import torch

        print_header("Model Inference", f"Using {model_type} model")
        console.print()

        if checkpoint is None:
            checkpoint = _find_best_checkpoint(model_type)
            if checkpoint is None:
                print_error(f"No checkpoint found for model type: {model_type}")
                print_info("Run 'fashionctl models list' to see available checkpoints")
                return

        checkpoint = Path(checkpoint)
        print_key_value("Checkpoint", str(checkpoint))
        print_key_value("Model type", model_type)
        print_key_value("Output", str(output_path))
        console.print()

        if not checkpoint.exists():
            print_error(f"Checkpoint not found: {checkpoint}")
            return

        is_visual = model_type == "visual" or "visual" in checkpoint.name

        load_kwargs = {"weights_only": False}

        if is_visual:
            from fashion.models.visual_lightning import VisualFashionRecommender

            print_info("Loading visual model...")
            model = VisualFashionRecommender.load_from_checkpoint(str(checkpoint), **load_kwargs)
        else:
            from fashion.models.lightning import FashionRecommender

            print_info("Loading model...")
            model = FashionRecommender.load_from_checkpoint(str(checkpoint), **load_kwargs)

        model.eval()

        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        model.to(device)
        print_key_value("Device", str(device))

        if is_visual:
            print_info("Running visual model inference...")
            _run_visual_inference(model, device, output_path, batch_size, sample_customers)
        else:
            from fashion.data.datamodule import FashionDataModule

            config_path = CONFIG_DIR / "infer.yaml"
            if config_path.exists():
                with initialize_config_dir(
                    config_dir=str(CONFIG_DIR.absolute()), version_base=None
                ):
                    cfg = compose(config_name="infer")
                datamodule = FashionDataModule(cfg.data)
            else:
                print_warning("No inference config found, using defaults")
                datamodule = FashionDataModule.for_inference(input_path, batch_size)

            datamodule.setup(stage="predict")
            predictions = []

            with console.status("[bold]Processing batches...[/bold]"), torch.no_grad():
                for batch in datamodule.predict_dataloader():
                    batch = {k: v.to(device) for k, v in batch.items()}
                    output = model.predict_step(batch, 0)
                    predictions.extend(output)

            results_df = pd.DataFrame(predictions)
            results_df.to_csv(output_path, index=False)

            print_success(f"Predictions saved to {output_path}")
            print_key_value("Total predictions", str(len(predictions)))

    def list(self) -> None:
        """List available model checkpoints."""
        print_header("Available Models")
        console.print()

        if not CHECKPOINTS_DIR.exists():
            print_warning("No checkpoints directory found")
            print_info("Train a model first with 'fashionctl models train'")
            return

        checkpoints = list(CHECKPOINTS_DIR.glob("*.ckpt"))
        if not checkpoints:
            print_warning("No checkpoints found")
            return

        table = create_table("Checkpoints", ["Name", "Size", "Modified"])

        for ckpt in sorted(checkpoints, key=lambda p: p.stat().st_mtime, reverse=True):
            size_mb = ckpt.stat().st_size / (1024 * 1024)
            mtime = datetime.datetime.fromtimestamp(ckpt.stat().st_mtime)
            table.add_row(ckpt.name, f"{size_mb:.1f} MB", mtime.strftime("%Y-%m-%d %H:%M"))

        console.print(table)
