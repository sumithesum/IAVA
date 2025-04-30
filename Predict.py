import os
from operator import index

import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet50
import torchvision.transforms as transforms
from VGG import VGG16
from Dataset import LandmarkTrainDataset, LandmarkTestDataset


def predict_image(image_path, model, transform, index_to_landmark, device):

    image = Image.open(image_path).convert("RGB")


    input_tensor = transform(image).unsqueeze(0).to(device)


    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        pred_idx = torch.argmax(output, dim=1).item()
        landmark_id = index_to_landmark[pred_idx]

    return landmark_id


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
train_df = pd.read_csv("train.csv").head(10000)
train_df.to_csv("train_10k.csv", index=False)

unique_landmarks = train_df['landmark_id'].unique()
landmark_id_to_index = {landmark_id: idx for idx, landmark_id in enumerate(unique_landmarks)}
num_classes = len(unique_landmarks)

train_dataset = LandmarkTrainDataset(csv_file="train_10k.csv", root_dir="train", transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

unique_landmarks = train_df["landmark_id"].unique()
landmark_id_to_index = {landmark_id: idx for idx, landmark_id in enumerate(unique_landmarks)}
index_to_landmark = {v: k for k, v in landmark_id_to_index.items()}


image_path = "Test.jpg"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 230)
model.load_state_dict(torch.load("Re17.pth"))
model = model.to(device)
model.eval()

# model = VGG16(num_classes=141)
# model.load_state_dict(torch.load("vgg16_landmark10k.pth"))
# model = model.to(device)
# model.eval()

landmark_pred = predict_image(image_path, model, test_transform, index_to_landmark, device)
print("Landmark prezis:", landmark_pred)

