import torch
from pydantic import BaseSettings


# Define a settings class for hyperparameters and paths
class TrainingSettings(BaseSettings):
    learning_rate: float = 0.0001
    num_epochs: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 32
    save_interval: int = 1
    train_cnn: bool = False
    spam_folder: str
    ham_folder: str

    # Ratio settings
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 1.0 - train_ratio - val_ratio

    class Config:
        env_file = ".env"
