from .utils import CustomImageCarDataset, load_dataset, training_model
from .model import build_model_classification
from ..utils import plot_loss_history, plot_metric_history


from pathlib import Path
import torch
import torch.nn as nn
from torch import optim 
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import argparse

current_file_path = Path(__file__)
project_dir = current_file_path.parent.parent.parent

# get the current learning rate helper function
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']

parser = argparse.ArgumentParser(description="Run Train Classification model.")
parser.add_argument(
    "--num_epochs",
    type=int,
    default=2,
    help="Number of epochs for training the model (default: 2)"
)
args = parser.parse_args()
# Defining epoch
epochs = args.num_epochs

source_dir = f'{project_dir}/data/classification'
df_train, df_val = load_dataset(source_dir)

class_names = {
    0: 'City Car',
    1: 'Big Truck',
    2: 'Multi Purpose Vehicle',
    3: 'Sedan',
    4: 'Sport Utility Vehicle',
    5: 'Truck',
    6: 'Van'
}
num_classes=len(class_names)

# Create Dataset 
train_dataset = CustomImageCarDataset(df_train, source_dir, None)
val_dataset = CustomImageCarDataset(df_val, source_dir, None)

# Create Data Loaders (training)
train_dl = DataLoader(train_dataset,
                      batch_size=32,
                      shuffle=True)

# Create Data Loader (validation)
val_dl = DataLoader(val_dataset,
                    batch_size=32,
                    shuffle=False)



model = build_model_classification(class_num=num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# defining loss function
loss_func = nn.CrossEntropyLoss(reduction='sum')

# defining optimiser
optimiser= optim.Adam(model.parameters(),
                 lr=1e-4)

current_lr = get_lr(optimiser)
print(f'current lr = {current_lr}')

# Define learning rate
lr_scheduler = CosineAnnealingLR(optimiser,
                                 T_max=2,
                                 eta_min=1e-5)

loss_history={"train": [],"val": []} # history of loss values in each epoch
metric_history={"train": [],"val": []} # histroy of metric values in each epoch
best_loss=float('inf') # initialize best loss to a large value

# Create directories for saving logs and best model if they don't exist
logs_dir = f"{project_dir}/logs/classification"
training_history_dir = f"{logs_dir}/training_history"
best_model_dir = f"{logs_dir}/best_model"

os.makedirs(logs_dir, exist_ok=True)
os.makedirs(training_history_dir, exist_ok=True)
os.makedirs(best_model_dir, exist_ok=True)

print(f"Ensured directories exist: {training_history_dir}, {best_model_dir}")


loss_history, metric_history = training_model(
    train_dl,
    val_dl,
    model,
    optimiser,
    device,
    num_epochs=2,
    lr_scheduler=lr_scheduler,
    loss_func=loss_func,
    path_to_save=f"{logs_dir}")

plot_loss_history(loss_history, training_history_dir)
plot_metric_history(metric_history, training_history_dir)

print('Finished Training')