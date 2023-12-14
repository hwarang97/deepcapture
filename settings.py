import torch
from pydantic_settings import BaseSettings


class TrainingSettings(BaseSettings):
    learning_rate: float = 0.0001
    num_epochs: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 32
    save_interval: int = 1
    train_cnn: bool = False
    test_cnn: bool = False
    train_xgb: bool = False
    test_xgb: bool = True
    spam_folder: str
    ham_folder: str
    target_spam_folder: str
    target_ham_folder: str
    credential_path: str
    augment_spam: bool = False
    augment_ham: bool = False
    nums_spam: int
    nums_ham: int
    cnn_model_path: str
    xgb_model_path: str

    # Ratio settings
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 1.0 - train_ratio - val_ratio

    class Config:
        env_file = ".env"


TrainingSettings = TrainingSettings()
