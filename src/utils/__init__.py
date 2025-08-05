"""
Main utilities for the fraud detection system.

This module contains helper classes and functions for configuration,
logging and common system operations.
"""

import yaml
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
from loguru import logger
import torch


@dataclass
class Config:
    """
    System configuration class.
    
    Loads and manages all project configurations from
    YAML files and environment variables.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = Path(config_path)
        self._config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configurations from YAML file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file: {e}")
            raise
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'model.hidden_dim')
            default: Default value if key doesn't exist
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value.
        
        Args:
            key: Configuration key
            value: New value
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    @property
    def data(self) -> Dict[str, Any]:
        """Return data configurations."""
        return self._config.get('data', {})
    
    @property
    def model(self) -> Dict[str, Any]:
        """Return model configurations."""
        return self._config.get('model', {})
    
    @property
    def training(self) -> Dict[str, Any]:
        """Return training configurations."""
        return self._config.get('training', {})
    
    @property
    def hardware(self) -> Dict[str, Any]:
        """Return hardware configurations."""
        return self._config.get('hardware', {})


class DeviceManager:
    """Device manager (CPU/GPU) for the system."""
    
    def __init__(self, device: str = "auto"):
        """
        Initialize device manager.
        
        Args:
            device: Desired device ('auto', 'cpu', 'cuda')
        """
        self.device = self._get_device(device)
        logger.info(f"Selected device: {self.device}")
    
    def _get_device(self, device: str) -> torch.device:
        """
        Determine appropriate device.
        
        Args:
            device: Requested device
            
        Returns:
            PyTorch device
        """
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        elif device == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                logger.warning("CUDA not available, using CPU")
                return torch.device("cpu")
        else:
            return torch.device("cpu")
    
    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to appropriate device."""
        return tensor.to(self.device)
    
    def get_device_info(self) -> Dict[str, Any]:
        """Return device information."""
        info = {
            "device": str(self.device),
            "cuda_available": torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            info.update({
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory": torch.cuda.get_device_properties(0).total_memory,
            })
        
        return info


class Logger:
    """Custom logging system."""
    
    def __init__(self, name: str = "fraud_detection", level: str = "INFO"):
        """
        Initialize logger.
        
        Args:
            name: Logger name
            level: Logging level
        """
        self.name = name
        self.level = level
        self._setup_logger()
    
    def _setup_logger(self) -> None:
        """Set up logging system."""
        # Remove default loguru handlers
        logger.remove()
        
        # Add console handler
        logger.add(
            lambda msg: print(msg, end=""),
            level=self.level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                   "<level>{message}</level>",
            colorize=True
        )
        
        # Add file handler
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logger.add(
            log_dir / f"{self.name}.log",
            rotation="1 day",
            retention="7 days",
            level=self.level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
        )
    
    def get_logger(self):
        """Return logger instance."""
        return logger


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object of the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_model(model: torch.nn.Module, path: Union[str, Path], 
               metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Save PyTorch model with metadata.
    
    Args:
        model: Model to be saved
        path: Path to save
        metadata: Additional metadata
    """
    path = Path(path)
    ensure_directory(path.parent)
    
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
    }
    
    if metadata:
        save_dict.update(metadata)
    
    torch.save(save_dict, path)
    logger.info(f"Model saved to {path}")


def load_model(model_class: torch.nn.Module, path: Union[str, Path]) -> torch.nn.Module:
    """
    Load PyTorch model.
    
    Args:
        model_class: Model class
        path: Path to saved model
        
    Returns:
        Loaded model
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    
    checkpoint = torch.load(path, map_location='cpu')
    model = model_class()
    model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info(f"Model loaded from {path}")
    return model


def format_metrics(metrics: Dict[str, float], precision: int = 4) -> str:
    """
    Format metrics for display.
    
    Args:
        metrics: Metrics dictionary
        precision: Number of decimal places
        
    Returns:
        Formatted string
    """
    formatted = []
    for key, value in metrics.items():
        if isinstance(value, float):
            formatted.append(f"{key}: {value:.{precision}f}")
        else:
            formatted.append(f"{key}: {value}")
    
    return " | ".join(formatted)


class Timer:
    """Context for measuring execution time."""
    
    def __init__(self, name: str = "Operation"):
        """
        Initialize timer.
        
        Args:
            name: Operation name
        """
        self.name = name
        self.start_time: Optional[float] = None
    
    def __enter__(self):
        """Start timer."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        """End timer and display result."""
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            logger.info(f"{self.name} executed in {elapsed:.2f}s")


# Global configuration instance
config = Config()
device_manager = DeviceManager(config.get('hardware.device', 'auto'))
system_logger = Logger()
