import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score

from extract_feature import extract_feature


def train_model(
    model, train_loader, val_loader, num_epochs, learning_rate, device, save_interval=1
):
    # loss func, optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # store value
    train_losses = []
    val_losses = []
    f1_scores = []
    best_loss = float("inf")

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs, _ = model(images)
            loss = criterion(outputs, labels.float().unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        train_loss = train_loss / len(train_loader.dataset)

        # valdiataion
        model.eval()
        valid_loss = 0.0
        valid_preds, valid_targets = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs, _ = model(images)
                loss = criterion(outputs, labels.float().unsqueeze(1))
                valid_loss += loss.item() * images.size(0)
                predicted = (outputs > 0.5).float().squeeze()
                valid_preds.extend(predicted.cpu().numpy().flatten())
                valid_targets.extend(labels.cpu().numpy())

            valid_loss = valid_loss / len(val_loader.dataset)
            f1 = f1_score(valid_targets, valid_preds, average="binary")

        print(
            f"Epoch {epoch+1}/{num_epochs}, \
            Train Loss: {train_loss:.4f}, \
            Valid Loss: {valid_loss:.4f}, \
            F1 Score: {f1:.4f}"
        )

        if epoch % save_interval == 0:
            if valid_loss < best_loss:
                best_loss = valid_loss
                torch.save(model.state_dict(), "model_checkpoint.pth")

    train_losses.append(train_loss)
    val_losses.append(valid_loss)
    f1_scores.append(f1)


def train_xgb(xgb_model, train_loader, val_loader, device):
    # extract
    features_train, labels_train = extract_feature(xgb_model, train_loader, device)
    features_val, labels_val = extract_feature(xgb_model, val_loader, device)

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

    # Evaluate
    predictions = xgb_model.predict(features_val)
    accuracy = accuracy_score(labels_val, predictions)
    print("Validation Accuracy: {:.2f}%".format(accuracy * 100))
