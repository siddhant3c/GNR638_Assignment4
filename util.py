import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class SmallCNN(nn.Module):
    def __init__(self, num_classes):
        super(SmallCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Compute the correct size after conv + pool layers
        self.flatten_size = self._get_flatten_size()
        
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def _get_flatten_size(self):
        # Create a dummy tensor to calculate output size dynamically
        with torch.no_grad():
            x = torch.zeros(1, 3, 224, 224)  # Example input size
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = F.relu(self.conv3(x))
            return x.numel()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten dynamically
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def explore_dataset(data_dir):
    categories = os.listdir(data_dir)
    category_distribution = {}
    print("Exploring dataset...")
    for category in tqdm(categories, desc="Categories"):
        category_path = os.path.join(data_dir, category)
        category_distribution[category] = len(os.listdir(category_path))
    return categories, category_distribution

def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    print("Training finished")

def get_cam(model, img_tensor, label):
    model.eval()
    features = model.conv3(model.pool(F.relu(model.conv2(model.pool(F.relu(model.conv1(img_tensor)))))))
    output = model.fc2(model.fc1(features.view(-1, 128 * 28 * 28)))
    
    weights = model.fc2.weight[label]
    cam = torch.sum(weights.unsqueeze(-1).unsqueeze(-1) * features, dim=1)
    cam = F.relu(cam)
    cam = F.interpolate(cam.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
    cam = cam.squeeze().detach().cpu().numpy()
    return cam