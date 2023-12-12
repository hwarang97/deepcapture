import argparse
from test import test_model

import numpy as np
import torch
import xgboost as xgb
from sklearn.metrics import accuracy_score

from data_loader import get_loaders, split_dataset
from extract_feature import extract_feature
from model import CNNModel
from train import train_model

# ratio setting
parser = argparse.ArgumentParser(description="Image Spam Detection Training")
parser.add_argument(
    "--train_ratio", type=float, default=0.8, help="Training data ratio (default: 0.8)"
)
parser.add_argument(
    "--val_ratio", type=float, default=0.1, help="Validation data ratio (default: 0.1)"
)
args = parser.parse_args()

train_ratio = args.train_ratio
val_ratio = args.val_ratio
test_ratio = 1.0 - train_ratio - val_ratio

# hyper-parameters
learning_rate = 0.0001
num_epochs = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
save_interval = 1
train_cnn = False
test_cnn = False

# path
spam_folder = "/mnt/c/Users/Kim Seok Je/Desktop/대학원/데이터보안과 프라이버시/report/personal_image_spam/personal_image_spam"
ham_folder = "/mnt/c/Users/Kim Seok Je/Desktop/대학원/데이터보안과 프라이버시/report/personal_image_ham/personal_image_ham"


def main():
    # split data
    spam_train, spam_val, spam_test = split_dataset(
        spam_folder, label=1, val_size=val_ratio, test_size=test_ratio
    )
    ham_train, ham_val, ham_test = split_dataset(
        ham_folder, label=0, val_size=val_ratio, test_size=test_ratio
    )

    # combine data
    train_data = spam_train + ham_train
    val_data = spam_val + ham_val
    test_data = spam_test + ham_test

    train_loader, val_loader, test_loader = get_loaders(
        train_data, val_data, test_data, batch_size=batch_size
    )

    # init modle
    model = CNNModel().to(device)
    xgb_model = xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=100,
        early_stopping_rounds=10,
        seed=123,
    )

    # train
    if train_cnn:
        # train, val
        train_model(model, train_loader, val_loader, num_epochs, learning_rate, device)

    # extract
    features_train, labels_train = extract_feature(model, train_loader, device)
    features_val, labels_val = extract_feature(model, val_loader, device)

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
    if test_cnn:
        test_model(model, test_loader, device)


if __name__ == "__main__":
    main()
