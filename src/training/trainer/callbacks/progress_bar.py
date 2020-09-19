"""Progress bar callback for the training loop."""
from typing import Dict, Optional

from tqdm import tqdm
from training.trainer.callbacks import Callback


class ProgressBar(Callback):
    """A TQDM progress bar for the training loop."""

    def __init__(self, epochs: int, log_batch_frequency: int = None) -> None:
        """Initializes the tqdm callback."""
        self.epochs = epochs
        print(epochs, type(epochs))
        self.log_batch_frequency = log_batch_frequency
        self.progress_bar = None
        self.val_metrics = {}

    def _configure_progress_bar(self) -> None:
        """Configures the tqdm progress bar with custom bar format."""
        self.progress_bar = tqdm(
            total=len(self.model.train_dataloader()),
            leave=False,
            unit="steps",
            mininterval=self.log_batch_frequency,
            bar_format="{desc} |{bar:32}| {n_fmt}/{total_fmt} ETA: {remaining} {rate_fmt}{postfix}",
        )

    def _key_abbreviations(self, logs: Dict) -> Dict:
        """Changes the length of keys, so that the progress bar fits better."""

        def rename(key: str) -> str:
            """Renames accuracy to acc."""
            return key.replace("accuracy", "acc")

        return {rename(key): value for key, value in logs.items()}

    # def on_fit_begin(self) -> None:
    #     """Creates a tqdm progress bar."""
    #     self._configure_progress_bar()

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict]) -> None:
        """Updates the description with the current epoch."""
        if epoch == 1:
            self._configure_progress_bar()
        else:
            self.progress_bar.reset()
        self.progress_bar.set_description(f"Epoch {epoch}/{self.epochs}")

    def on_epoch_end(self, epoch: int, logs: Dict) -> None:
        """At the end of each epoch, the validation metrics are updated to the progress bar."""
        self.val_metrics = logs
        self.progress_bar.set_postfix(**self._key_abbreviations(logs))
        self.progress_bar.update()

    def on_train_batch_end(self, batch: int, logs: Dict) -> None:
        """Updates the progress bar for each training step."""
        if self.val_metrics:
            logs.update(self.val_metrics)
        self.progress_bar.set_postfix(**self._key_abbreviations(logs))
        self.progress_bar.update()

    def on_fit_end(self) -> None:
        """Closes the tqdm progress bar."""
        self.progress_bar.close()
