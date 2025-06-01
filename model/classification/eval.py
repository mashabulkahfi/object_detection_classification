from .model import build_model_classification
from .utils import CustomImageCarDataset, loss_epoch, process_df
from ..utils import load_model

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

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
task_name = 'classification'

current_file_path = Path(__file__)
project_dir = current_file_path.parent.parent.parent

source_dir = f'{project_dir}/data/classification'

df_test = process_df(source_dir, 'test')
test_dataset = CustomImageCarDataset(df_test, source_dir, None)

# Create Data Loader (validation)
test_dl = DataLoader(test_dataset,
                    batch_size=32,
                    shuffle=False)


model = build_model_classification(class_num=num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


load_model(model, f"{project_dir}/logs/{task_name}/best_model/best_model_weights.pth", device)
model = model.to(device)

model.eval()
loss_fn = nn.CrossEntropyLoss()

test_loss, test_metric = loss_epoch(model, loss_fn, test_dl, sanity_check=False, opt=None, device=device)

print(f"Test Loss: {test_loss}, Test Metric: {test_metric}")