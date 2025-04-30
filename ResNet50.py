import torch
import torch.nn as nn
import torchvision.models as models
import pandas as pd
import torch.optim as optim
import glob

from matplotlib import pyplot as plt
from torch.utils.data import random_split

from Submission import generate_submission
from epoch import train_one_epoch, evaluate_one_epoch
import torchvision.transforms as transforms
from Dataset import LandmarkTrainDataset, LandmarkTestDataset
from torch.utils.data import DataLoader
import os

def get_model(num_classes):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

if __name__ == "__main__":

    train_df = pd.read_csv("train.csv").head(20000)
    train_df.to_csv("train_10k.csv", index=False)

    unique_landmarks = train_df['landmark_id'].unique()
    landmark_id_to_index = {landmark_id: idx for idx, landmark_id in enumerate(unique_landmarks)}
    num_classes = len(unique_landmarks)
    print("Numar clase:", num_classes)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = LandmarkTrainDataset(csv_file="train_10k.csv", root_dir="train", transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)


    test_paths = glob.glob('test/*/*/*/*.jpg')


    test_ids = [os.path.splitext(os.path.basename(p))[0] for p in test_paths]
    test_dataset = LandmarkTestDataset(test_ids, root_dir="test", transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()


    val_size = int(0.2 * len(train_dataset))
    train_size = len(train_dataset) - val_size

    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    print("Număr imagini test:", len(test_loader.dataset))


    num_epochs = 17

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate_one_epoch(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoca {epoch + 1}/{num_epochs} — Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    torch.save(model.state_dict(), "Re" + str(num_epochs)+".pth")
    print("Model salvat")

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, marker='o', label='Train Loss')

    plt.plot(range(1, num_epochs + 1), val_losses, marker='x', label='Validation Loss')

    plt.title("Evoluția loss-ului pe epoci")
    plt.xlabel("Epoca")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Salvăm imaginea
    plt.savefig("training_loss_plot" + str(num_epochs) +".png")
    plt.close()

    index_to_landmark = {v: k for k, v in train_dataset.landmark_id_to_index.items()}

    generate_submission(model, test_loader, index_to_landmark, device)