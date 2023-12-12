from test import test_model

import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score

from data_loader import get_loaders, split_dataset
from extract_feature import extract_feature
from model import CNNModel
from settings import TrainingSettings as Ts
from train import train_model


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

    # extract
    features_train, labels_train = extract_feature(model, train_loader, Ts.device)
    features_val, labels_val = extract_feature(model, val_loader, Ts.device)

    # list to numpy
    features_train = np.array(features_train)
    labels_train = np.array(labels_train)
    features_val = np.array(features_val)
    labels_val = np.array(labels_val)

    # train, val
    xgb_model.fit(
        features_train,
        labels_train,
        eval_set=[(features_train, labels_train), (features_val, labels_val)],
    )
    xgb_model.save_model("xgb_model.json")

    predictions = xgb_model.predict(features_val)
    accuracy = accuracy_score(labels_val, predictions)
    print("Validation Accuracy: {:.2f}%".format(accuracy * 100))

    # test
    if Ts.test_cnn:
        test_model(model, test_loader, Ts.device)


if __name__ == "__main__":
    main()
