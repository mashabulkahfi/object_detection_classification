from pathlib import Path
import torch

from torch.utils.data import DataLoader

from .model import build_model_object_detection
from ..utils import load_model
from .utils import loss_epoch, load_dataset, CarDataset, collate_fn, Averager

current_file_path = Path(__file__)
project_dir = current_file_path.parent.parent.parent

DIR_TEST = f"{project_dir}/data/detection/test"

df_images_test, df_annot_test_filter  = load_dataset(DIR_TEST)

test_dataset = CarDataset(df_images_test, df_annot_test_filter, DIR_TEST, None)

test_data_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=2,
    collate_fn=collate_fn
)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = build_model_object_detection(backbone='resnet50', num_class=2, use_pretrained=False)
model_save_path = f'{project_dir}/logs/detection/best_model/best_model_weights.pth'
load_model(model, model_save_path, device)
model.to(device)

print(f"Model weights loaded successfully from {model_save_path}")

test_loss_tracker = Averager()

test_loss = loss_epoch(process_name="Testing",
                       loss_tracker=test_loss_tracker, 
                       data_loader=test_data_loader, 
                       model = model,
                       optimizer=None,
                       device=device, 
                       path_to_save=None
                    )
print(f"Test Loss: {test_loss}")