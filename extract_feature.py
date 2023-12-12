import torch


def extract_feature(model, data_loader, device):
    # Load the model
    model.load_state_dict(torch.load("model_checkpoint.pth"))
    model.eval()

    features = []
    labels_list = []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            # extract feature
            _, feature = model(images)

            features.extend(feature.cpu().tolist())
            labels_list.extend(labels.cpu().tolist())

    return features, labels_list
