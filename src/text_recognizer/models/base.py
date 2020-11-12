"""Abstract Model class for PyTorch neural networks."""

from abc import ABC, abstractmethod
from glob import glob
import importlib
from pathlib import Path
import re
import shutil
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from loguru import logger
import torch
from torch import nn
from torch import Tensor
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import DataLoader, Dataset, random_split
from torchsummary import summary
from torchvision.transforms import Compose

from text_recognizer.datasets import EmnistMapper

WEIGHT_DIRNAME = Path(__file__).parents[1].resolve() / "weights"


class Model(ABC):
    """Abstract Model class with composition of different parts defining a PyTorch neural network."""

    def __init__(
        self,
        network_fn: Type[nn.Module],
        dataset: Type[Dataset],
        network_args: Optional[Dict] = None,
        dataset_args: Optional[Dict] = None,
        metrics: Optional[Dict] = None,
        criterion: Optional[Callable] = None,
        criterion_args: Optional[Dict] = None,
        optimizer: Optional[Callable] = None,
        optimizer_args: Optional[Dict] = None,
        lr_scheduler: Optional[Callable] = None,
        lr_scheduler_args: Optional[Dict] = None,
        swa_args: Optional[Dict] = None,
        device: Optional[str] = None,
    ) -> None:
        """Base class, to be inherited by model for specific type of data.

        Args:
            network_fn (Type[nn.Module]): The PyTorch network.
            dataset (Type[Dataset]): A dataset class.
            network_args (Optional[Dict]): Arguments for the network. Defaults to None.
            dataset_args (Optional[Dict]):  Arguments for the dataset.
            metrics (Optional[Dict]): Metrics to evaluate the performance with. Defaults to None.
            criterion (Optional[Callable]): The criterion to evaluate the performance of the network.
                Defaults to None.
            criterion_args (Optional[Dict]): Dict of arguments for criterion. Defaults to None.
            optimizer (Optional[Callable]): The optimizer for updating the weights. Defaults to None.
            optimizer_args (Optional[Dict]): Dict of arguments for optimizer. Defaults to None.
            lr_scheduler (Optional[Callable]): A PyTorch learning rate scheduler. Defaults to None.
            lr_scheduler_args (Optional[Dict]): Dict of arguments for learning rate scheduler. Defaults to
                None.
            swa_args (Optional[Dict]): Dict of arguments for stochastic weight averaging. Defaults to
                None.
            device (Optional[str]): Name of the device to train on. Defaults to None.

        """
        # Has to be set in subclass.
        self._mapper = None

        # Placeholder.
        self._input_shape = None

        self.dataset = dataset
        self.dataset_args = dataset_args

        # Placeholders for datasets.
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # Stochastic Weight Averaging placeholders.
        self.swa_args = swa_args
        self._swa_scheduler = None
        self._swa_network = None
        self._use_swa_model = False

        # Experiment directory.
        self.model_dir = None

        # Flag for configured model.
        self.is_configured = False
        self.data_prepared = False

        # Flag for stopping training.
        self.stop_training = False

        self._name = (
            f"{self.__class__.__name__}_{dataset.__name__}_{network_fn.__name__}"
        )

        self._metrics = metrics if metrics is not None else None

        # Set the device.
        self._device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device is None
            else device
        )

        # Configure network.
        self._network = None
        self._network_args = network_args
        self._configure_network(network_fn)

        # Place network on device (GPU).
        self.to_device()

        # Loss and Optimizer placeholders for before loading.
        self._criterion = criterion
        self.criterion_args = criterion_args

        self._optimizer = optimizer
        self.optimizer_args = optimizer_args

        self._lr_scheduler = lr_scheduler
        self.lr_scheduler_args = lr_scheduler_args

    def configure_model(self) -> None:
        """Configures criterion and optimizers."""
        if not self.is_configured:
            self._configure_criterion()
            self._configure_optimizers()

            # Set this flag to true to prevent the model from configuring again.
            self.is_configured = True

    def _configure_transforms(self) -> None:
        # Load transforms.
        transforms_module = importlib.import_module(
            "text_recognizer.datasets.transforms"
        )
        if (
            "transform" in self.dataset_args["args"]
            and self.dataset_args["args"]["transform"] is not None
        ):
            transform_ = []
            for t in self.dataset_args["args"]["transform"]:
                args = t["args"] or {}
                transform_.append(getattr(transforms_module, t["type"])(**args))
            self.dataset_args["args"]["transform"] = Compose(transform_)

        if (
            "target_transform" in self.dataset_args["args"]
            and self.dataset_args["args"]["target_transform"] is not None
        ):
            target_transform_ = [
                torch.tensor,
            ]
            for t in self.dataset_args["args"]["target_transform"]:
                args = t["args"] or {}
                target_transform_.append(getattr(transforms_module, t["type"])(**args))
            self.dataset_args["args"]["target_transform"] = Compose(target_transform_)

    def prepare_data(self) -> None:
        """Prepare data for training."""
        # TODO add downloading.
        if not self.data_prepared:
            self._configure_transforms()

            # Load train dataset.
            train_dataset = self.dataset(train=True, **self.dataset_args["args"])
            train_dataset.load_or_generate_data()

            # Set input shape.
            self._input_shape = train_dataset.input_shape

            # Split train dataset into a training and validation partition.
            dataset_len = len(train_dataset)
            train_len = int(
                self.dataset_args["train_args"]["train_fraction"] * dataset_len
            )
            val_len = dataset_len - train_len
            self.train_dataset, self.val_dataset = random_split(
                train_dataset, lengths=[train_len, val_len]
            )

            # Load test dataset.
            self.test_dataset = self.dataset(train=False, **self.dataset_args["args"])
            self.test_dataset.load_or_generate_data()

            # Set the flag to true to disable ability to load data agian.
            self.data_prepared = True

    def train_dataloader(self) -> DataLoader:
        """Returns data loader for training set."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.dataset_args["train_args"]["batch_size"],
            num_workers=self.dataset_args["train_args"]["num_workers"],
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Returns data loader for validation set."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.dataset_args["train_args"]["batch_size"],
            num_workers=self.dataset_args["train_args"]["num_workers"],
            shuffle=True,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Returns data loader for test set."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.dataset_args["train_args"]["batch_size"],
            num_workers=self.dataset_args["train_args"]["num_workers"],
            shuffle=False,
            pin_memory=True,
        )

    def _configure_network(self, network_fn: Type[nn.Module]) -> None:
        """Loads the network."""
        # If no network arguments are given, load pretrained weights if they exist.
        if self._network_args is None:
            self.load_weights(network_fn)
        else:
            self._network = network_fn(**self._network_args)

    def _configure_criterion(self) -> None:
        """Loads the criterion."""
        self._criterion = (
            self._criterion(**self.criterion_args)
            if self._criterion is not None
            else None
        )

    def _configure_optimizers(self,) -> None:
        """Loads the optimizers."""
        if self._optimizer is not None:
            self._optimizer = self._optimizer(
                self._network.parameters(), **self.optimizer_args
            )
        else:
            self._optimizer = None

        if self._optimizer and self._lr_scheduler is not None:
            if "steps_per_epoch" in self.lr_scheduler_args:
                self.lr_scheduler_args["steps_per_epoch"] = len(self.train_dataloader())

            # Assume lr scheduler should update at each epoch if not specified.
            if "interval" not in self.lr_scheduler_args:
                interval = "epoch"
            else:
                interval = self.lr_scheduler_args.pop("interval")
            self._lr_scheduler = {
                "lr_scheduler": self._lr_scheduler(
                    self._optimizer, **self.lr_scheduler_args
                ),
                "interval": interval,
            }

        if self.swa_args is not None:
            self._swa_scheduler = {
                "swa_scheduler": SWALR(self._optimizer, swa_lr=self.swa_args["lr"]),
                "swa_start": self.swa_args["start"],
            }
            self._swa_network = AveragedModel(self._network).to(self.device)

    @property
    def name(self) -> str:
        """Returns the name of the model."""
        return self._name

    @property
    def input_shape(self) -> Tuple[int, ...]:
        """The input shape."""
        return self._input_shape

    @property
    def mapper(self) -> EmnistMapper:
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
    def lr_scheduler(self) -> Optional[Dict]:
        """Returns a directory with the learning rate scheduler."""
        return self._lr_scheduler

    @property
    def swa_scheduler(self) -> Optional[Dict]:
        """Returns a directory with the stochastic weight averaging scheduler."""
        return self._swa_scheduler

    @property
    def swa_network(self) -> Optional[Callable]:
        """Returns the stochastic weight averaging network."""
        return self._swa_network

    @property
    def network(self) -> Type[nn.Module]:
        """Neural network."""
        # Returns the SWA network if available.
        return self._network

    @property
    def weights_filename(self) -> str:
        """Filepath to the network weights."""
        WEIGHT_DIRNAME.mkdir(parents=True, exist_ok=True)
        return str(WEIGHT_DIRNAME / f"{self._name}_weights.pt")

    def use_swa_model(self) -> None:
        """Set to use predictions from SWA model."""
        if self.swa_network is not None:
            self._use_swa_model = True

    def forward(self, x: Tensor) -> Tensor:
        """Feedforward pass with the network."""
        if self._use_swa_model:
            return self.swa_network(x)
        else:
            return self.network(x)

    def summary(
        self,
        input_shape: Optional[Union[List, Tuple]] = None,
        depth: int = 4,
        device: Optional[str] = None,
    ) -> None:
        """Prints a summary of the network architecture."""
        device = self.device if device is None else device

        if input_shape is not None:
            summary(self.network, input_shape, depth=depth, device=device)
        elif self._input_shape is not None:
            input_shape = (1,) + tuple(self._input_shape)
            summary(self.network, input_shape, depth=depth, device=device)
        else:
            logger.warning("Could not print summary as input shape is not set.")

    def to_device(self) -> None:
        """Places the network on the device (GPU)."""
        self._network.to(self._device)

    def _get_state_dict(self) -> Dict:
        """Get the state dict of the model."""
        state = {"model_state": self._network.state_dict()}

        if self._optimizer is not None:
            state["optimizer_state"] = self._optimizer.state_dict()

        if self._lr_scheduler is not None:
            state["scheduler_state"] = self._lr_scheduler["lr_scheduler"].state_dict()
            state["scheduler_interval"] = self._lr_scheduler["interval"]

        if self._swa_network is not None:
            state["swa_network"] = self._swa_network.state_dict()

        return state

    def load_from_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """Load a previously saved checkpoint.

        Args:
            checkpoint_path (Path): Path to the experiment with the checkpoint.

        """
        checkpoint_path = Path(checkpoint_path)
        self.prepare_data()
        self.configure_model()
        logger.debug("Loading checkpoint...")
        if not checkpoint_path.exists():
            logger.debug("File does not exist {str(checkpoint_path)}")

        checkpoint = torch.load(str(checkpoint_path), map_location=self.device)
        self._network.load_state_dict(checkpoint["model_state"])

        if self._optimizer is not None:
            self._optimizer.load_state_dict(checkpoint["optimizer_state"])

        if self._lr_scheduler is not None:
            # Does not work when loading from previous checkpoint and trying to train beyond the last max epochs
            # with OneCycleLR.
            if self._lr_scheduler["lr_scheduler"].__class__.__name__ != "OneCycleLR":
                self._lr_scheduler["lr_scheduler"].load_state_dict(
                    checkpoint["scheduler_state"]
                )
                self._lr_scheduler["interval"] = checkpoint["scheduler_interval"]

        if self._swa_network is not None:
            self._swa_network.load_state_dict(checkpoint["swa_network"])

    def save_checkpoint(
        self, checkpoint_path: Path, is_best: bool, epoch: int, val_metric: str
    ) -> None:
        """Saves a checkpoint of the model.

        Args:
            checkpoint_path (Path): Path to the experiment with the checkpoint.
            is_best (bool): If it is the currently best model.
            epoch (int): The epoch of the checkpoint.
            val_metric (str): Validation metric.

        """
        state = self._get_state_dict()
        state["is_best"] = is_best
        state["epoch"] = epoch
        state["network_args"] = self._network_args

        checkpoint_path.mkdir(parents=True, exist_ok=True)

        logger.debug("Saving checkpoint...")
        filepath = str(checkpoint_path / "last.pt")
        torch.save(state, filepath)

        if is_best:
            logger.debug(
                f"Found a new best {val_metric}. Saving best checkpoint and weights."
            )
            shutil.copyfile(filepath, str(checkpoint_path / "best.pt"))

    def load_weights(self, network_fn: Type[nn.Module]) -> None:
        """Load the network weights."""
        logger.debug("Loading network with pretrained weights.")
        filename = glob(self.weights_filename)[0]
        if not filename:
            raise FileNotFoundError(
                f"Could not find any pretrained weights at {self.weights_filename}"
            )
        # Loading state directory.
        state_dict = torch.load(filename, map_location=torch.device(self._device))
        self._network_args = state_dict["network_args"]
        weights = state_dict["model_state"]

        # Initializes the network with trained weights.
        self._network = network_fn(**self._network_args)
        self._network.load_state_dict(weights)

        if "swa_network" in state_dict:
            self._swa_network = AveragedModel(self._network).to(self.device)
            self._swa_network.load_state_dict(state_dict["swa_network"])

    def save_weights(self, path: Path) -> None:
        """Save the network weights."""
        logger.debug("Saving the best network weights.")
        shutil.copyfile(str(path / "best.pt"), self.weights_filename)
