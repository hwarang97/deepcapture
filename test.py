import numpy as np
import torch
import torch.nn as nn
from extract_feature import extract_feature
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


def test_model(model, test_loader, device, model_path):
    # Load the model
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # criterion
    criterion = nn.BCELoss()

    # eval on test set
    test_loss = 0.0
    test_preds, test_targets = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs, _ = model(images)
            loss = criterion(outputs, labels.float().unsqueeze(1))
            test_loss += loss.item() * images.size(0)
            predicted = (outputs > 0.5).float().squeeze()
            test_preds.extend(predicted.cpu().numpy())
            test_targets.extend(labels.cpu().numpy())

    test_loss = test_loss / len(test_loader.dataset)
    f1 = f1_score(test_targets, test_preds, average="binary")

    print(f"Test Loss: {test_loss:.4f}, F1 Score: {f1:.4f}")


def test_xgb(model, xgb_model, test_loader, device, xgb_model_path):
    # Load model
    xgb_model.load_model(xgb_model_path)

    # extract
    features_test, labels_test = extract_feature(model, test_loader, device)

    # list to numpy
    features_test = np.array(features_test)
    labels_test = np.array(labels_test)

    # test
    predictions = xgb_model.predict(features_test)

    # matrics
    accuracy = accuracy_score(labels_test, predictions)
    precision = precision_score(labels_test, predictions)
    recall = recall_score(labels_test, predictions)
    f1 = f1_score(labels_test, predictions)

    print(f"Accuracy: {accuracy:.4f}%")
    print(f"Precision: {precision:.4f}%")
    print(f"Recall: {recall:.4f}%")
    print(f"F1 Score: {f1:.4f}%")
