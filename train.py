# CS688W-Systems for Deep Learning
# Assignment -1 exploring VGG6 on CIFAR-10 with different configuration
# cs24m534 Santi Bhusan Acharya 

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from model import VGG6
from dataloader import get_dataloaders

activation_map = {
    "ReLU": nn.ReLU,
    "Sigmoid": nn.Sigmoid,
    "Tanh": nn.Tanh,
    "SiLU": nn.SiLU,
    "GELU": nn.GELU
}

optimizer_map = {
    "SGD": lambda params, lr: optim.SGD(params, lr=lr),
    "Nesterov": lambda params, lr: optim.SGD(params, lr=lr, momentum=0.9, nesterov=True),
    "Adam": lambda params, lr: optim.Adam(params, lr=lr),
    "Adagrad": lambda params, lr: optim.Adagrad(params, lr=lr),
    "RMSprop": lambda params, lr: optim.RMSprop(params, lr=lr),
    "Nadam": lambda params, lr: optim.NAdam(params, lr=lr)
}

def train():
    # Initialize W&B with custom run name
    wandb.init(project="vgg6-sweep-sba10")
    config = wandb.config
    wandb.run.name = f"act_{config.activation}_bs_{config.batch_size}_opt_{config.optimizer}_lr_{config.learning_rate}_ep_{config.epochs}"
    
    # Setup model and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGG6(activation_map[config.activation]).to(device)
    optimizer = optimizer_map[config.optimizer](model.parameters(), lr = config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    trainloader, testloader = get_dataloaders(config.batch_size)

    best_acc = 0.0

    # Training loop
    for epoch in range(config.epochs):
        model.train()
        correct = total = train_loss = 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        #train_acc = 100. * model.eval()
        model.eval()
        correct = total = val_loss = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        train_acc = 100. * correct / total
        val_acc = 100. * correct / total
        val_loss /= len(testloader)

         # Log metrics
        wandb.log({
            "Epoch": epoch + 1,
            "Train Accuracy": train_acc,
            "Train Loss": train_loss,
            "Validation Accuracy": val_acc,
            "Validation Loss": val_loss
        })

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")

            # Upload to W&B
            artifact = wandb.Artifact("best_model", type="model")
            artifact.add_file("best_model.pth")
            wandb.log_artifact(artifact)

if __name__ == "__main__":
    sweep_config = {
        'method': 'random',
        'metric': {'name': 'Validation Accuracy', 'goal': 'maximize'},
        'parameters': {
            'activation': {'values': ['ReLU', 'Sigmoid', 'Tanh', 'SiLU', 'GELU']},
            'optimizer': {'values': ['SGD', 'Nesterov', 'Adam', 'Adagrad', 'RMSprop', 'Nadam']},
            'batch_size': {'values': [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]},
            'epochs': {'values': [20, 40, 60, 80, 100]},
            'learning_rate': {'values': [0.01, 0.001, 0.0001]}
        }
    }
    sweep_id = wandb.sweep(sweep_config, project="vgg6-sweep-sba10")
    wandb.agent(sweep_id, function=train)
