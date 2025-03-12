import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from PIL import Image
from tqdm import tqdm

from util import *

# Define the dataset directory
data_dir = 'data/train'

# Explore the dataset
categories, category_distribution = explore_dataset(data_dir)
print("Categories:", categories)
print("Category distribution:", category_distribution)

# Data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
print("Loading datasets...")
train_dataset = datasets.ImageFolder(root='data/train', transform=transform)
test_dataset = datasets.ImageFolder(root='data/test', transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize the model, loss function, and optimizer
model = SmallCNN(num_classes=len(categories))

# Check if GPU is available and use it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print("\nUsing device:", device)
model.to(device)

optimizers = {
    'SGD': optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
    'Adam': optim.Adam(model.parameters(), lr=0.001),
    'RMSprop': optim.RMSprop(model.parameters(), lr=0.001)
}

criterion = nn.CrossEntropyLoss()

loss_history = {key: [] for key in optimizers.keys()}

num_epochs = 10


if __name__ == "__main__":
    for opt_name, optimizer in optimizers.items():
        print(f"\n------------------\nTraining with {opt_name} optimizer")
        model = SmallCNN(num_classes=len(categories)).to(device)  # Reinitialize model for each optimizer
        
        for epoch in tqdm(range(num_epochs)):
            model.train()
            epoch_loss = 0
            
            for inputs, labels in tqdm(train_loader, desc=f"{opt_name} Epoch {epoch+1}/{num_epochs}"):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_epoch_loss = epoch_loss / len(train_loader)
            loss_history[opt_name].append(avg_epoch_loss)
            print(f"{opt_name} Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}")


    plt.figure(figsize=(10, 6))
    for opt_name, losses in loss_history.items():
        plt.plot(range(1, num_epochs + 1), losses, label=opt_name)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Convergence Comparison Across Optimizers')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('loss_convergence.png')
    plt.show()