
"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import torch
import data_loader, model, engine, utils

import torchvision.transforms.v2 as transforms
import argparse

# -----------------------------------------------------------------

parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=True)

parser.add_argument(
        "--epochs",
        default=12,
        type=int,
        help="number of epochs" )

parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="batch size")

parser.add_argument(
        "--hidden_units",
        default=10,
        type=int,
        help="total hidden units")

parser.add_argument(
        "--lr",
        default=0.001,
        type=float,
        help="learning rate")

args = parser.parse_args()

#--------------------------------------------------------------------

# Setup hyperparameters
NUM_EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
HIDDEN_UNITS = args.hidden_units
LEARNING_RATE = args.lr

# Setup directories
basedir = "../"
train_dir = basedir + "data/pizza_steak_sushi/train"
test_dir = basedir + "data/pizza_steak_sushi/test"

# Setup target device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create transforms
train_transform_trivial_augment = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToImage(), 
    transforms.ToDtype(torch.float32, scale=True)
])

# Create testing transform (no data augmentation)
# test_transform = transforms.Compose([
#     transforms.Resize((64, 64)),
#     transforms.ToImage(), 
#     transforms.ToDtype(torch.float32, scale=True)
# ])

# Create DataLoaders
train_dataloader, test_dataloader, class_names = data_loader.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=train_transform_trivial_augment,
    batch_size=BATCH_SIZE
)

# Model, loss, optimizer
model = model.TinyVGG( input_shape=3, hidden_units=HIDDEN_UNITS, output_shape=len(class_names))
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Start training with help from engine.py
engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device)

# Save the model with help from utils.py
utils.save_model(model=model, target_dir="../models", model_name="tinyvgg_model_v1.pth")

