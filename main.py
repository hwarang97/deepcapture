from test import test_model

import xgboost as xgb

from data_loader import get_loaders, split_dataset
from model import CNNModel
from settings import TrainingSettings as Ts
from train import train_model, train_xgb


def main():
    # split data
    spam_train, spam_val, spam_test = split_dataset(
        Ts.spam_folder, label=1, val_size=Ts.val_ratio, test_size=Ts.test_ratio
    )
    ham_train, ham_val, ham_test = split_dataset(
        Ts.ham_folder, label=0, val_size=Ts.val_ratio, test_size=Ts.test_ratio
    )

    # combine data
    train_data = spam_train + ham_train
    val_data = spam_val + ham_val
    test_data = spam_test + ham_test

    train_loader, val_loader, test_loader = get_loaders(
        train_data, val_data, test_data, batch_size=Ts.batch_size
    )

    # init modle
    model = CNNModel().to(Ts.device)
    xgb_model = xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=100,
        early_stopping_rounds=10,
        seed=123,
    )

    # train
    if Ts.train_cnn:
        # train, val
        train_model(
            model, train_loader, val_loader, Ts.num_epochs, Ts.learning_rate, Ts.device
        )

    train_xgb(model, xgb_model, train_loader, val_loader, Ts.device)

    # test
    if Ts.test_cnn:
        test_model(model, test_loader, Ts.device)


if __name__ == "__main__":
    main()
