# CS688W-Systems for Deep Learning
# Assignment -1 exploring VGG6 on CIFAR-10 with different configuration
# cs24m534 Santi Bhusan Acharya 

import torch
import torch.nn as nn
from model import VGG6
from dataloader import get_dataloaders

activation_fn = nn.ReLU
model = VGG6(activation_fn)
#model.load_state_dict(torch.load("best_model.pth"))
model.load_state_dict(torch.load("best_model.pth", map_location=torch.device('cpu')))
model.eval()

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')
model.to(device)

_, testloader = get_dataloaders(batch_size=100)
correct = total = 0
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

test_acc = 100. * correct / total
print(f"Test Accuracy of the best model: {test_acc:.2f}%")
