
import os

import torch

import torch.nn.functional as F

from torch.utils.data import DataLoader
from pathlib import Path

from .utils import collate_fn, CarDataset, Averager, load_dataset, training_model
from .model import build_model_object_detection
from ..utils import plot_loss_history
import argparse

parser = argparse.ArgumentParser(description="Run Train Classification model.")
parser.add_argument(
    "--num_epochs",
    type=int,
    default=2,
    help="Number of epochs for training the model (default: 2)"
)
args = parser.parse_args()

current_file_path = Path(__file__)
project_dir = current_file_path.parent.parent.parent

DIR_TRAIN = f"{project_dir}/data/detection/train"
DIR_VALID = f"{project_dir}/data/detection/valid"

num_epochs = args.num_epochs

# Dataset Preparation

df_images_train, df_annot_train_filter  = load_dataset(DIR_TRAIN)
df_images_valid, df_annot_valid_filter = load_dataset(DIR_VALID)

train_dataset = CarDataset(df_images_train, df_annot_train_filter, DIR_TRAIN, None)
valid_dataset = CarDataset(df_images_valid, df_annot_valid_filter, DIR_VALID, None)

train_data_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    num_workers=2,
    collate_fn=collate_fn
)

valid_data_loader = DataLoader(
    valid_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=2,
    collate_fn=collate_fn
)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Training the model
model = build_model_object_detection(backbone='resnet50', num_class=2, use_pretrained=True)

model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
lr_scheduler = None

# Create directories for saving logs and best model if they don't exist
logs_dir = f"{project_dir}/logs/detection"
training_history_dir = f"{logs_dir}/training_history"
best_model_dir = f"{logs_dir}/best_model"

os.makedirs(logs_dir, exist_ok=True)
os.makedirs(training_history_dir, exist_ok=True)
os.makedirs(best_model_dir, exist_ok=True)

print(f"Ensured directories exist: {training_history_dir}, {best_model_dir}")

training_loss_tracker = Averager()
validation_loss_tracker = Averager()

loss_history = training_model(train_data_loader,
                   training_loss_tracker,
                   valid_data_loader,
                   validation_loss_tracker,
                   model,
                   optimizer,
                   device,
                   num_epochs,
                   lr_scheduler=lr_scheduler,
                   path_to_save=logs_dir)

plot_loss_history(loss_history, training_history_dir)