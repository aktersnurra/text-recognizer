"""GPUManager class."""
import os
import time
from typing import Optional

import gpustat
from loguru import logger
import numpy as np
from redlock import Redlock


GPU_LOCK_TIMEOUT = 5000  # ms


class GPUManager:
    """Class for allocating GPUs."""

    def __init__(self, verbose: bool = False) -> None:
        """Initializes Redlock manager."""
        self.lock_manager = Redlock([{"host": "localhost", "port": 6379, "db": 0}])
        self.verbose = verbose

    def get_free_gpu(self) -> int:
        """Gets a free GPU.

        If some GPUs are available, try reserving one by checking out an exclusive redis lock.
        If none available or can not get lock, sleep and check again.

        Returns:
            int: The gpu index.

        """
        while True:
            gpu_index = self._get_free_gpu()
            if gpu_index is not None:
                return gpu_index

            if self.verbose:
                logger.debug(f"pid {os.getpid()} sleeping")
            time.sleep(GPU_LOCK_TIMEOUT / 1000)

    def _get_free_gpu(self) -> Optional[int]:
        """Fetches an available GPU index."""
        try:
            available_gpu_indices = [
                gpu.index
                for gpu in gpustat.GPUStatCollection.new_query()
                if gpu.memory_used < 0.5 * gpu.memory_total
            ]
        except Exception as e:
            logger.debug(f"Got the following exception: {e}")
            return None

        if available_gpu_indices:
            gpu_index = np.random.choice(available_gpu_indices)
            if self.verbose:
                logger.debug(f"pid {os.getpid()} picking gpu {gpu_index}")
            if self.lock_manager.lock(f"gpu_{gpu_index}", GPU_LOCK_TIMEOUT):
                return int(gpu_index)
            if self.verbose:
                logger.debug(f"pid {os.getpid()} could not get lock.")
        return None
