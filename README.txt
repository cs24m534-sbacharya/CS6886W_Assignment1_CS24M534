
# VGG6 CIFAR-10 Deep Learning Project

This project explores training a custom VGG6 model on the CIFAR-10 dataset using PyTorch. It is modularized into separate files for clarity and maintainability.

## 📁 File Structure

- `model.py`: Contains the definition of the VGG6 neural network architecture.
- `dataloader.py`: Handles data loading and preprocessing for CIFAR-10.
- `train.py`: Trains the model using various hyperparameter configurations and saves the best model.
- `test.py`: Loads the best saved model and evaluates it on the test set.

## 🚀 How to Use

### 1. Install Dependencies
Make sure you have the required packages installed:
```bash
pip install torch torchvision wandb
```

### 2. Train the Model
Run the training script to start a W&B sweep and train the model:
```bash
python train.py
```
This will save the best model as `best_model.pth`.

### 3. Test the Model
Evaluate the saved model on the test set:
```bash
python test.py
```

## 🧠 Notes
- The model uses different activation functions and optimizers as part of the sweep.
- W&B is used to log metrics and manage hyperparameter sweeps.
- CIFAR-10 dataset is automatically downloaded and preprocessed.

