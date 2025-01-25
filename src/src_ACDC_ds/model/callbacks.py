import shutil
from pathlib import Path
from transformers import TrainerCallback

class CheckpointCallback(TrainerCallback):
    """Custom callback for checkpoint management."""

    def __init__(self, save_dir: Path, save_steps: int = 100, max_checkpoints: int = 3):
        self.save_dir = save_dir
        self.save_steps = save_steps
        self.max_checkpoints = max_checkpoints
        self.checkpoints = []

    def on_step_end(self, args, state, control, **kwargs):
        """Save checkpoint on step end if needed."""
        if state.global_step % self.save_steps == 0:
            checkpoint_path = self.save_dir / f"checkpoint-{state.global_step}"

            kwargs['trainer'].save_model(checkpoint_path)

            self.checkpoints.append(checkpoint_path)

            if len(self.checkpoints) > self.max_checkpoints:
                oldest_checkpoint = self.checkpoints.pop(0)
                if oldest_checkpoint.exists():
                    shutil.rmtree(oldest_checkpoint)

class CustomEarlyStoppingCallback(TrainerCallback):
    """Custom early stopping callback with more flexibility."""

    def __init__(
        self,
        metric_name: str = "eval_dice_mean",
        patience: int = 5,
        min_improvement: float = 0.01,
        min_epochs: int = 10
    ):
        self.metric_name = metric_name
        self.patience = patience
        self.min_improvement = min_improvement
        self.min_epochs = min_epochs
        self.best_metric = None
        self.no_improvement_count = 0

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        """Check for early stopping conditions after evaluation."""
        if state.epoch < self.min_epochs:
            return

        current_metric = metrics.get(self.metric_name)
        if current_metric is None:
            return

        if self.best_metric is None:
            self.best_metric = current_metric
        elif current_metric > self.best_metric + self.min_improvement:
            self.best_metric = current_metric
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

        if self.no_improvement_count >= self.patience:
            control.should_training_stop = True
            print(f"\nEarly stopping triggered after {state.epoch} epochs")
            print(f"Best {self.metric_name}: {self.best_metric:.4f}")