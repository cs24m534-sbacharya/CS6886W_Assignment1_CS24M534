# CS688W-Systems for Deep Learning
# Assignment -1 exploring VGG6 on CIFAR-10 with different configuration
# Santi Bhusan Acharya 

!pip install wandb

import wandb
wandb.login()

# Sweep Configuration
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

#Training Function
def train():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader

    # Initialize W&B with custom run name
    
    wandb.init(project="vgg6-sweep-sba8")
    wandb.run.name = f"act_{wandb.config.activation}_bs_{wandb.config.batch_size}_opt_{wandb.config.optimizer}_lr_{wandb.config.learning_rate}_ep_{wandb.config.epochs}"
    #wandb.run.save()
    config = wandb.config


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

    class VGG6(nn.Module):
        def __init__(self, activation_fn):
            super(VGG6, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                activation_fn(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                activation_fn(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                activation_fn(),
                nn.MaxPool2d(2, 2),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256 * 4 * 4, 512),
                activation_fn(),
                nn.Linear(512, 10)
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    # Data transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    # Load data
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    trainloader = DataLoader(trainset, batch_size=config.batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=100, shuffle=False)

    # Setup model and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGG6(activation_map[config.activation]).to(device)
    #optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer = optimizer_map[config.optimizer](model.parameters(), lr = config.learning_rate)
    criterion = nn.CrossEntropyLoss()

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

# Launch the sweep
sweep_id = wandb.sweep(sweep_config, project="vgg6-sweep-sba8")
wandb.agent(sweep_id, function=train)