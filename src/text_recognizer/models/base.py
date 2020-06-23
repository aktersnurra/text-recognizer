"""Abstract Model class for PyTorch neural networks."""

from abc import ABC, abstractmethod
from pathlib import Path
import shutil
from typing import Callable, Dict, Optional, Tuple

from loguru import logger
import torch
from torch import nn
from torchsummary import summary

from text_recognizer.dataset.data_loader import fetch_data_loader

WEIGHT_DIRNAME = Path(__file__).parents[1].resolve() / "weights"


class Model(ABC):
    """Abstract Model class with composition of different parts defining a PyTorch neural network."""

    def __init__(
        self,
        network_fn: Callable,
        network_args: Dict,
        data_loader_args: Optional[Dict] = None,
        metrics: Optional[Dict] = None,
        criterion: Optional[Callable] = None,
        criterion_args: Optional[Dict] = None,
        optimizer: Optional[Callable] = None,
        optimizer_args: Optional[Dict] = None,
        lr_scheduler: Optional[Callable] = None,
        lr_scheduler_args: Optional[Dict] = None,
        device: Optional[str] = None,
    ) -> None:
        """Base class, to be inherited by predictors for specific type of data.

        Args:
            network_fn (Callable): The PyTorch network.
            network_args (Dict): Arguments for the network.
            data_loader_args (Optional[Dict]):  Arguments for the data loader.
            metrics (Optional[Dict]): Metrics to evaluate the performance with. Defaults to None.
            criterion (Optional[Callable]): The criterion to evaulate the preformance of the network.
                Defaults to None.
            criterion_args (Optional[Dict]): Dict of arguments for criterion. Defaults to None.
            optimizer (Optional[Callable]): The optimizer for updating the weights. Defaults to None.
            optimizer_args (Optional[Dict]): Dict of arguments for optimizer. Defaults to None.
            lr_scheduler (Optional[Callable]): A PyTorch learning rate scheduler. Defaults to None.
            lr_scheduler_args (Optional[Dict]): Dict of arguments for learning rate scheduler. Defaults to
                None.
            device (Optional[str]): Name of the device to train on. Defaults to None.

        """

        # Fetch data loaders.
        if data_loader_args is not None:
            self._data_loaders = fetch_data_loader(**data_loader_args)
            dataset_name = self._data_loaders.items()[0].dataset.__name__
        else:
            dataset_name = ""
            self._data_loaders = None

        self.name = f"{self.__class__.__name__}_{dataset_name}_{network_fn.__name__}"

        # Extract the input shape for the torchsummary.
        self._input_shape = network_args.pop("input_shape")

        if metrics is not None:
            self._metrics = metrics

        # Set the device.
        if self.device is None:
            self._device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )
        else:
            self._device = device

        # Load network.
        self._network = network_fn(**network_args)

        # To device.
        self._network.to(self._device)

        # Set criterion.
        self._criterion = None
        if criterion is not None:
            self._criterion = criterion(**criterion_args)

        # Set optimizer.
        self._optimizer = None
        if optimizer is not None:
            self._optimizer = optimizer(self._network.parameters(), **optimizer_args)

        # Set learning rate scheduler.
        self._lr_scheduler = None
        if lr_scheduler is not None:
            self._lr_scheduler = lr_scheduler(self._optimizer, **lr_scheduler_args)

    @property
    def input_shape(self) -> Tuple[int, ...]:
        """The input shape."""
        return self._input_shape

    def eval(self) -> None:
        """Sets the network to evaluation mode."""
        self._network.eval()

    def train(self) -> None:
        """Sets the network to train mode."""
        self._network.train()

    @property
    def device(self) -> str:
        """Device where the weights are stored, i.e. cpu or cuda."""
        return self._device

    @property
    def metrics(self) -> Optional[Dict]:
        """Metrics."""
        return self._metrics

    @property
    def criterion(self) -> Optional[Callable]:
        """Criterion."""
        return self._criterion

    @property
    def optimizer(self) -> Optional[Callable]:
        """Optimizer."""
        return self._optimizer

    @property
    def lr_scheduler(self) -> Optional[Callable]:
        """Learning rate scheduler."""
        return self._lr_scheduler

    @property
    def data_loaders(self) -> Optional[Dict]:
        """Dataloaders."""
        return self._data_loaders

    @property
    def network(self) -> nn.Module:
        """Neural network."""
        return self._network

    @property
    def weights_filename(self) -> str:
        """Filepath to the network weights."""
        WEIGHT_DIRNAME.mkdir(parents=True, exist_ok=True)
        return str(WEIGHT_DIRNAME / f"{self.name}_weights.pt")

    def summary(self) -> None:
        """Prints a summary of the network architecture."""
        summary(self._network, self._input_shape, device=self.device)

    def _get_state(self) -> Dict:
        """Get the state dict of the model."""
        state = {"model_state": self._network.state_dict()}
        if self._optimizer is not None:
            state["optimizer_state"] = self._optimizer.state_dict()
        return state

    def load_checkpoint(self, path: Path) -> int:
        """Load a previously saved checkpoint.

        Args:
            path (Path): Path to the experiment with the checkpoint.

        Returns:
            epoch (int): The last epoch when the checkpoint was created.

        """
        if not path.exists():
            logger.debug("File does not exist {str(path)}")

        checkpoint = torch.load(str(path))
        self._network.load_state_dict(checkpoint["model_state"])

        if self._optimizer is not None:
            self._optimizer.load_state_dict(checkpoint["optimizer_state"])

        epoch = checkpoint["epoch"]

        return epoch

    def save_checkpoint(
        self, path: Path, is_best: bool, epoch: int, val_metric: str
    ) -> None:
        """Saves a checkpoint of the model.

        Args:
            path (Path): Path to the experiment folder.
            is_best (bool): If it is the currently best model.
            epoch (int): The epoch of the checkpoint.
            val_metric (str): Validation metric.

        """
        state = self._get_state_dict()
        state["is_best"] = is_best
        state["epoch"] = epoch

        path.mkdir(parents=True, exist_ok=True)

        logger.debug("Saving checkpoint...")
        filepath = str(path / "last.pt")
        torch.save(state, filepath)

        if is_best:
            logger.debug(
                f"Found a new best {val_metric}. Saving best checkpoint and weights."
            )
            self.save_weights()
            shutil.copyfile(filepath, str(path / "best.pt"))

    def load_weights(self) -> None:
        """Load the network weights."""
        logger.debug("Loading network weights.")
        weights = torch.load(self.weights_filename)["model_state"]
        self._network.load_state_dict(weights)

    def save_weights(self) -> None:
        """Save the network weights."""
        logger.debug("Saving network weights.")
        torch.save({"model_state": self._network.state_dict()}, self.weights_filename)

    @abstractmethod
    def mapping(self) -> Dict:
        """Mapping from network output to class."""
        ...
