from test import test_model
from test import test_xgb

import ham_augmentation as ha
import spam_augmentation as sa
import xgboost as xgb
from data_loader import get_image_paths_with_labels
from data_loader import get_loaders
from data_loader import split_dataset
from model import CNNModel
from settings import TrainingSettings as Ts
from train import train_model
from train import train_xgb


def main():
    # split data
    spam_train, spam_val, spam_test = split_dataset(
        Ts.spam_folder, label=1, val_size=Ts.val_ratio, test_size=Ts.test_ratio
    )
    ham_train, ham_val, ham_test = split_dataset(
        Ts.ham_folder, label=0, val_size=Ts.val_ratio, test_size=Ts.test_ratio
    )

    spam_test = get_image_paths_with_labels(
        "/mnt/c/Users/Kim Seok Je/Desktop/대학원/데이터보안과 프라이버시/term project/dredze-personal_image_spam/personal_image_spam",
        1,
    )
    ham_test = get_image_paths_with_labels(
        "/mnt/c/Users/Kim Seok Je/Desktop/대학원/데이터보안과 프라이버시/term project/dredze_personal_image_ham/personal_image_ham",
        0,
    )

    # augmentation
    if Ts.augment_spam:
        spam_train = sa.create_augmented_images(
            spam_train, Ts.target_spam_folder, Ts.nums_spam
        )

    if Ts.augment_ham:
        ham_train = ha.create_augmented_images(
            ham_train,
            Ts.target_ham_folder,
            Ts.nums_ham,
            Ts.credential_path,
        )

    spam_aug_images = get_image_paths_with_labels(Ts.target_spam_folder, 1)
    ham_aug_images = get_image_paths_with_labels(Ts.target_ham_folder, 1)
    spam_train = spam_train + spam_aug_images
    ham_train = ham_train + ham_aug_images

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
    )

    # train
    if Ts.train_cnn:
        # train, val
        train_model(
            model,
            train_loader,
            val_loader,
            Ts.num_epochs,
            Ts.learning_rate,
            Ts.device,
            Ts.cnn_model_path,
        )

    if Ts.train_xgb:
        train_xgb(
            model, xgb_model, train_loader, val_loader, Ts.device, Ts.xgb_model_path
        )

    # test
    if Ts.test_cnn:
        test_model(model, test_loader, Ts.device, Ts.cnn_model_path)

    if Ts.test_xgb:
        test_xgb(model, xgb_model, test_loader, Ts.device, Ts.xgb_model_path)


if __name__ == "__main__":
    main()
