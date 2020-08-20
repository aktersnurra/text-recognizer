"""Abstract Model class for PyTorch neural networks."""

from abc import ABC, abstractmethod
from glob import glob
from pathlib import Path
import re
import shutil
from typing import Callable, Dict, Optional, Tuple, Type

from loguru import logger
import torch
from torch import nn
from torchsummary import summary

from text_recognizer.datasets import EmnistMapper, fetch_data_loaders

WEIGHT_DIRNAME = Path(__file__).parents[1].resolve() / "weights"


class Model(ABC):
    """Abstract Model class with composition of different parts defining a PyTorch neural network."""

    def __init__(
        self,
        network_fn: Type[nn.Module],
        network_args: Optional[Dict] = None,
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
        """Base class, to be inherited by model for specific type of data.

        Args:
            network_fn (Type[nn.Module]): The PyTorch network.
            network_args (Optional[Dict]): Arguments for the network. Defaults to None.
            data_loader_args (Optional[Dict]):  Arguments for the DataLoader.
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

        # Configure data loaders and dataset info.
        dataset_name, self._data_loaders, self._mapper = self._configure_data_loader(
            data_loader_args
        )
        self._input_shape = self._mapper.input_shape

        self._name = f"{self.__class__.__name__}_{dataset_name}_{network_fn.__name__}"

        if metrics is not None:
            self._metrics = metrics

        # Set the device.
        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = device

        # Configure network.
        self._network, self._network_args = self._configure_network(
            network_fn, network_args
        )

        # To device.
        self._network.to(self._device)

        # Configure training objects.
        self._criterion = self._configure_criterion(criterion, criterion_args)
        self._optimizer, self._lr_scheduler = self._configure_optimizers(
            optimizer, optimizer_args, lr_scheduler, lr_scheduler_args
        )

        # Experiment directory.
        self.model_dir = None

        # Flag for stopping training.
        self.stop_training = False

    def _configure_data_loader(
        self, data_loader_args: Optional[Dict]
    ) -> Tuple[str, Dict, EmnistMapper]:
        """Loads data loader, dataset name, and dataset mapper."""
        if data_loader_args is not None:
            data_loaders = fetch_data_loaders(**data_loader_args)
            dataset = list(data_loaders.values())[0].dataset
            dataset_name = dataset.__name__
            mapper = dataset.mapper
        else:
            self._mapper = EmnistMapper()
            dataset_name = "*"
            data_loaders = None
        return dataset_name, data_loaders, mapper

    def _configure_network(
        self, network_fn: Type[nn.Module], network_args: Optional[Dict]
    ) -> Tuple[Type[nn.Module], Dict]:
        """Loads the network."""
        # If no network arguemnts are given, load pretrained weights if they exist.
        if network_args is None:
            network, network_args = self.load_weights(network_fn)
        else:
            network = network_fn(**network_args)
        return network, network_args

    def _configure_criterion(
        self, criterion: Optional[Callable], criterion_args: Optional[Dict]
    ) -> Optional[Callable]:
        """Loads the criterion."""
        if criterion is not None:
            _criterion = criterion(**criterion_args)
        else:
            _criterion = None
        return _criterion

    def _configure_optimizers(
        self,
        optimizer: Optional[Callable],
        optimizer_args: Optional[Dict],
        lr_scheduler: Optional[Callable],
        lr_scheduler_args: Optional[Dict],
    ) -> Tuple[Optional[Callable], Optional[Callable]]:
        """Loads the optimizers."""
        if optimizer is not None:
            _optimizer = optimizer(self._network.parameters(), **optimizer_args)
        else:
            _optimizer = None

        if self._optimizer and lr_scheduler is not None:
            if "OneCycleLR" in str(lr_scheduler):
                lr_scheduler_args["steps_per_epoch"] = len(self._data_loaders["train"])
            _lr_scheduler = lr_scheduler(self._optimizer, **lr_scheduler_args)
        else:
            _lr_scheduler = None

        return _optimizer, _lr_scheduler

    @property
    def __name__(self) -> str:
        """Returns the name of the model."""
        return self._name

    @property
    def input_shape(self) -> Tuple[int, ...]:
        """The input shape."""
        return self._input_shape

    @property
    def mapper(self) -> Dict:
        """Returns the mapper that maps between ints and chars."""
        return self._mapper

    @property
    def mapping(self) -> Dict:
        """Returns the mapping between network output and Emnist character."""
        return self._mapper.mapping

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
        return str(WEIGHT_DIRNAME / f"{self._name}_weights.pt")

    def summary(self) -> None:
        """Prints a summary of the network architecture."""
        device = re.sub("[^A-Za-z]+", "", self.device)
        if self._input_shape is not None:
            input_shape = (1,) + tuple(self._input_shape)
            summary(self._network, input_shape, device=device)
        else:
            logger.warning("Could not print summary as input shape is not set.")

    def _get_state_dict(self) -> Dict:
        """Get the state dict of the model."""
        state = {"model_state": self._network.state_dict()}

        if self._optimizer is not None:
            state["optimizer_state"] = self._optimizer.state_dict()

        if self._lr_scheduler is not None:
            state["scheduler_state"] = self._lr_scheduler.state_dict()

        return state

    def load_checkpoint(self, path: Path) -> int:
        """Load a previously saved checkpoint.

        Args:
            path (Path): Path to the experiment with the checkpoint.

        Returns:
            epoch (int): The last epoch when the checkpoint was created.

        """
        logger.debug("Loading checkpoint...")
        if not path.exists():
            logger.debug("File does not exist {str(path)}")

        checkpoint = torch.load(str(path))
        self._network.load_state_dict(checkpoint["model_state"])

        if self._optimizer is not None:
            self._optimizer.load_state_dict(checkpoint["optimizer_state"])

        # Does not work when loadning from previous checkpoint and trying to train beyond the last max epochs.
        # if self._lr_scheduler is not None:
        #     self._lr_scheduler.load_state_dict(checkpoint["scheduler_state"])

        epoch = checkpoint["epoch"]

        return epoch

    def save_checkpoint(self, is_best: bool, epoch: int, val_metric: str) -> None:
        """Saves a checkpoint of the model.

        Args:
            is_best (bool): If it is the currently best model.
            epoch (int): The epoch of the checkpoint.
            val_metric (str): Validation metric.

        Raises:
            ValueError: If the self.model_dir is not set.

        """
        state = self._get_state_dict()
        state["is_best"] = is_best
        state["epoch"] = epoch
        state["network_args"] = self._network_args

        if self.model_dir is None:
            raise ValueError("Experiment directory is not set.")

        self.model_dir.mkdir(parents=True, exist_ok=True)

        logger.debug("Saving checkpoint...")
        filepath = str(self.model_dir / "last.pt")
        torch.save(state, filepath)

        if is_best:
            logger.debug(
                f"Found a new best {val_metric}. Saving best checkpoint and weights."
            )
            shutil.copyfile(filepath, str(self.model_dir / "best.pt"))

    def load_weights(self, network_fn: Type[nn.Module]) -> Tuple[Type[nn.Module], Dict]:
        """Load the network weights."""
        logger.debug("Loading network with pretrained weights.")
        filename = glob(self.weights_filename)[0]
        if not filename:
            raise FileNotFoundError(
                f"Could not find any pretrained weights at {self.weights_filename}"
            )
        # Loading state directory.
        state_dict = torch.load(filename, map_location=torch.device(self._device))
        network_args = state_dict["network_args"]
        weights = state_dict["model_state"]

        # Initializes the network with trained weights.
        network = network_fn(**self._network_args)
        network.load_state_dict(weights)
        return network, network_args

    def save_weights(self, path: Path) -> None:
        """Save the network weights."""
        logger.debug("Saving the best network weights.")
        shutil.copyfile(str(path / "best.pt"), self.weights_filename)
