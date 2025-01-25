import torch
import wandb
import os
from pathlib import Path
from transformers import TrainingArguments, Trainer
from src.src_ACDC_ds.data.data_manipulation import create_dataloaders
from src.src_ACDC_ds.evaluation.metrics_and_loss import CombinedLoss, compute_detailed_metrics
from src.src_ACDC_ds.model.fine_tuning_model import CardiacSegmenter
from src.src_ACDC_ds.model.callbacks import CheckpointCallback, CustomEarlyStoppingCallback

class TrainingConfig:
  """Configuration for cardiac segmentation training."""
  # Model params
  num_classes: int = 4
  pretrained: bool = True
  freeze_backbone: bool = False

  # Training params
  output_dir: str = "cardiac_segmentation_checkpoints"
  num_train_epochs: int = 50
  batch_size: int = 8
  learning_rate: float = 2e-4
  weight_decay: float = 0.01
  warmup_ratio: float = 0.1
  save_steps: int = 100

  # Dataset params
  data_dir: Path = Path('/content/Med-Physics/data/ACDC_preprocessed/ACDC_training_slices/')
  target_size: tuple = (256, 256)
  num_workers: int = 4

  # Trainer params
  gradient_accumulation_steps: int = 2
  fp16: bool = True  # Mixed precision training
  logging_steps: int = 10

def get_training_args(config: TrainingConfig) -> TrainingArguments:
    """Get HuggingFace training arguments."""
    return TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        lr_scheduler_type="linear",
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        fp16=config.fp16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=config.logging_steps,
        save_total_limit=3,  # Keep only the last 3 checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="dice_mean",
        greater_is_better=True,
        report_to="tensorboard"
    )

class CardiacSegmentationTrainer(Trainer):
    """Custom trainer for segmentation tasks with detailed logging."""

    def __init__(self, metrics_fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = CombinedLoss(num_classes=4)
        self.metrics_fn = metrics_fn

    def compute_loss(self, model, inputs, return_outputs=False):
        """Custom loss computation."""
        labels = inputs.pop("labels")
        outputs = model(inputs["pixel_values"])
        loss = self.loss_fn(outputs, labels)
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs):
        """Custom training step with detailed logging."""
        loss = self.compute_loss(model, inputs)

        with torch.no_grad():
            outputs = model(inputs["pixel_values"])
            metrics = self.metrics_fn(outputs, inputs["labels"])

        if self.args.logging_steps > 0 and self.state.global_step % self.args.logging_steps == 0:
            self.log({
                "train/loss": loss.item(),
                **{f"train/{k}": v for k, v in metrics.items()},
                "train/learning_rate": self.get_learning_rate(),
                "train/epoch": self.state.epoch,
                "train/step": self.state.global_step,
            })

        return loss

    def evaluation_loop(self, *args, **kwargs):
        """Custom evaluation loop with detailed logging."""
        output = super().evaluation_loop(*args, **kwargs)

        metrics = self.metrics_fn(
            torch.from_numpy(output.predictions),
            torch.from_numpy(output.label_ids)
        )

        self.log({
            "eval/loss": output.metrics["eval_loss"],
            **{f"eval/{k}": v for k, v in metrics.items()},
            "eval/epoch": self.state.epoch,
            "eval/step": self.state.global_step,
        })

        output.metrics.update(metrics)
        return output

def setup_training(
    config: TrainingConfig,
    enable_logging: bool = False,
    project_name: str = "cardiac-segmentation"
) -> CardiacSegmentationTrainer:
    """
    Setup all components for training.

    Args:
        config: Training configuration
        enable_logging: Whether to enable wandb logging
        project_name: Name of the wandb project

    Returns:
        Configured trainer
    """
    try:
        print("1. Setting up logging...")
        if enable_logging:
            wandb.init(
                project=project_name,
                config={
                    "architecture": "SegFormer-B0",
                    "dataset": "ACDC",
                    "learning_rate": config.learning_rate,
                    "epochs": config.num_train_epochs,
                    "batch_size": config.batch_size,
                    "image_size": config.target_size,
                }
            )

        print("2. Creating dataloaders...")
        train_loader, val_loader = create_dataloaders(
            data_dir=config.data_dir,
            batch_size=config.batch_size,
            num_workers=config.num_workers
        )
        print(f"Created dataloaders with {len(train_loader.dataset)} training and "
              f"{len(val_loader.dataset)} validation samples")

        print("3. Creating model...")
        model = CardiacSegmenter(
            num_classes=config.num_classes,
            pretrained=config.pretrained,
            freeze_backbone=config.freeze_backbone
        )

        print("4. Setting up training arguments...")
        training_args = TrainingArguments(
            output_dir=config.output_dir,
            num_train_epochs=config.num_train_epochs,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            lr_scheduler_type="linear",
            weight_decay=config.weight_decay,
            warmup_ratio=config.warmup_ratio,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            fp16=config.fp16,
            evaluation_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="dice_mean",
            greater_is_better=True,
            logging_dir=os.path.join(config.output_dir, "logs"),
            logging_strategy="steps",
            logging_steps=10,
            report_to="wandb" if enable_logging else "none"
        )

        callbacks = [
        CheckpointCallback(
            save_dir=Path(config.output_dir),
            save_steps=config.save_steps,
            max_checkpoints=3
        ),
        CustomEarlyStoppingCallback(
            metric_name="eval_dice_mean",
            patience=5,
            min_improvement=0.01,
            min_epochs=10
        )]

        print("5. Creating trainer...")
        trainer = CardiacSegmentationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_loader.dataset,
        eval_dataset=val_loader.dataset,
        metrics_fn=compute_detailed_metrics,
        callbacks=callbacks
        )

        print("Training setup completed successfully!")
        return trainer

    except Exception as e:
        print(f"Error during training setup: {str(e)}")
        raise

def train_model(config: TrainingConfig):
    """
    Complete training pipeline.
    """

    # Setup training
    trainer = setup_training(config)

    # Training
    print("Starting training...")
    train_results = trainer.train()

    # Save final model
    trainer.save_model(config.output_dir / "final_model")

    return trainer, train_results