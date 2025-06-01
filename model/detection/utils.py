import json
import pandas as pd
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset

from ..utils import save_model, writing_training_logs

def filter_car_data(df_annot):
    """Filter the annotations DataFrame to include only car-related categories."""
    df_annot = df_annot[(df_annot['category_id'] == 3) | (df_annot['category_id'] == 6)]
    return df_annot

def read_annotation(dir_name, text_name):
    """Read annotations from a JSON file and return a DataFrame."""
    with open(f'{dir_name}/_annotations.coco.json', 'r') as file:
        data = json.load(file)
        df = pd.DataFrame(data[text_name])
    return df
  
def collate_fn(batch):
    """Collate function to combine a list of samples into a batch."""
    return tuple(zip(*batch))

def handle_bbox_format(df):
    """Handle the bounding box format in the DataFrame."""
    splited_bbox = df['bbox'].apply(pd.Series)
    #change columns
    splited_bbox.columns =['x','y','w','h']
    df = pd.concat([df,splited_bbox], axis=1)

    df = df.drop(['bbox','segmentation'], axis=1)
    df['x'] = df['x'].astype(np.float64)
    df['y'] = df['y'].astype(np.float64)
    df['w'] = df['w'].astype(np.float64)
    df['h'] = df['h'].astype(np.float64)

    return df

def load_dataset(dir_name):
    """Load the dataset from the specified directory."""
    df_categories = read_annotation(dir_name, 'categories')
    df_images = read_annotation(dir_name, 'images')
    df_annot = read_annotation(dir_name, 'annotations')

    df_annot = handle_bbox_format(df_annot)
    df_annot_filter = filter_car_data(df_annot)

    return df_images, df_annot_filter

class CarDataset(Dataset):

    def __init__(self, dataframe_image, dataframe_annot, image_dir, transforms=None):
        super().__init__()

        self.image_ids = dataframe_annot['image_id'].unique()
        self.df_image = dataframe_image
        self.df_annot = dataframe_annot
        self.image_dir = image_dir
        self.transforms = transforms

    def __getitem__(self, index: int):

        image_id = self.image_ids[index]
        # file_name = self.df_image[self.df_image['id'] == image_id]['file_name'].item()
        file_name = self.df_image[self.df_image['id'] == image_id]['file_name'].iloc[0]

        # Get all the bbox records for the current image_id
        records = self.df_annot[self.df_annot['image_id'] == image_id]

        # Read the image, and convert it to RGB format
        # Normalize the image to [0, 1] range
        image = cv2.imread(f'{self.image_dir}/{file_name}', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0


        #Change the bboxe format from xywh to xyxy
        boxes = records[['x', 'y', 'w', 'h']].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        # Calculate the area of the bounding boxes
        # area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = records['area'].values
        area = torch.as_tensor(area, dtype=torch.float32)

        # there is only one class
        # set the labels to 1 for each bounding box
        # records.shape[0] gives the number of bounding boxes for each image
        labels = torch.ones((records.shape[0],), dtype=torch.int64)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # suppose all instances are not crowd
        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([index])
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transforms:
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels
            }
            sample = self.transforms(**sample)
            image = sample['image']

            # Ensure transformed boxes are correctly handled and converted back to a tensor
            # We need to convert this back to a tensor of shape (N, 4)
            if sample['bboxes']: # Check if bboxes is not empty
                target['boxes'] = torch.tensor(sample['bboxes'], dtype=torch.float32)
                target['labels'] = torch.tensor(sample['labels'], dtype=torch.int64)

            else:
                # Handle cases where transforms might result in no bounding boxes
                target['boxes'] = torch.empty((0, 4), dtype=torch.float32)
                target['labels'] = torch.empty((0,), dtype=torch.int64)
                target['area'] = torch.empty((0,), dtype=torch.float32)
                target['iscrowd'] = torch.empty((0,), dtype=torch.int64)

        return image, target, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]

class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0

def loss_epoch(process_name="Training", 
               loss_tracker=None,
               data_loader=None,
               model=None,
               optimizer=None,
               device=None,
               path_to_save=None):
    """
    Function to compute the loss for a single epoch.
    Args:
        process_name (str): Name of the process, either "Training" or "Validation".
        loss_tracker (Averager): Tracker to accumulate loss values.
        data_loader (DataLoader): DataLoader for the dataset.   
        model (torch.nn.Module): The model to compute the loss.
        optimizer (torch.optim.Optimizer): Optimizer for the model.
        device (torch.device): Device to run the model on (CPU or GPU).
        path_to_save (str): Path to save the training logs.
    Returns:
        loss_value (float): The average loss value for the epoch.
    """
    
    if path_to_save is not None:
        path_batches_logs = f'{path_to_save}/training_history/{process_name.lower()}_loss_batches.txt'
    
    for itr, (images, targets, image_ids) in enumerate(data_loader):
        images = [torch.from_numpy(image).permute(2, 0, 1).to(device) for image in images]
        targets = [{k: torch.as_tensor(v).to(device) if isinstance(v, np.ndarray) else v.to(device) if hasattr(v, 'to') else v
        for k, v in t.items()} for t in targets]

        # Updated  
        if process_name != "Training":
            with torch.no_grad():
                model.train()  # Set the model in training mode to compute the loss
                loss_dict = model(images, targets)

                model.eval() # Set the model to evaluation mode after computing the loss
        else:
            model.train() # Set training mode for the model to compute the loss
            loss_dict = model(images, targets)
        
        if isinstance(loss_dict, dict):
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            loss_tracker.send(loss_value)

            #Only run this when training
            if process_name == "Training":
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                # Clear CUDA cache to help free up memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # TODO : CREATE A FUNCTION TO LOGS THE TRAINING LOGS
            if (itr + 1) % 50 == 0:
                print(f"{process_name} Iteration #{(itr + 1)} loss: {loss_value}")
                if path_to_save is not None:
                    writing_training_logs(f'{path_batches_logs}', 
                                      f"Iteration #{(itr + 1)} loss: {loss_value:.4f}")
    
    #Loss value after processing all batches
    loss_value = loss_tracker.value

    return loss_value

def training_model(train_data_loader,
                   training_loss_tracker,
                   valid_data_loader,
                   validation_loss_tracker,
                   model,
                   optimizer,
                   device,
                   num_epochs,
                   lr_scheduler=None,
                   path_to_save=None):
    """
    Function to train the model for a specified number of epochs.
    Args:
        train_data_loader (DataLoader): DataLoader for the training dataset.
        training_loss_tracker (Averager): Tracker to accumulate training loss values.
        valid_data_loader (DataLoader): DataLoader for the validation dataset.
        validation_loss_tracker (Averager): Tracker to accumulate validation loss values.
        model (torch.nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): Optimizer for the model.
        device (torch.device): Device to run the model on (CPU or GPU).
        num_epochs (int): Number of epochs to train the model.
        lr_scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler. Defaults to None.
        path_to_save (str, optional): Path to save training logs and best model weights. Defaults to None.
    Returns:
        loss_history (dict): Dictionary containing training and validation loss history.
    """
    loss_history = {"train": [], "val": []}  # history of loss values in each epoch

    best_validation_loss = float('inf')  # Initialize best validation loss to a large value

    path_training_logs = f'{path_to_save}/training_history/training_logs.txt'
    path_best_model = f'{path_to_save}/best_model/best_model_weights.pth'

    for epoch in range(num_epochs):
        training_loss_tracker.reset() # Reset the loss tracker for each epoch
        validation_loss_tracker.reset() # Reset the validation loss tracker for each epoch

        #path_to_save = root_dir + '/logs/detection'
        writing_training_logs(f'{path_training_logs}', 
                              f"Starting Epoch #{epoch+1}...")
        print(f"Starting training for Epoch #{epoch+1}...")

        training_loss = loss_epoch(
            process_name="Training", 
            loss_tracker=training_loss_tracker,
            data_loader=train_data_loader,
            model=model,
            optimizer=optimizer,
            device=device,
            path_to_save=path_to_save
        )

        # update the learning rate
        if lr_scheduler is not None:
            lr_scheduler.step()
        
        loss_history["train"].append(training_loss)

        print(f"Epoch #{epoch+1} Training loss: {training_loss}")
        writing_training_logs(f'{path_training_logs}', 
                              f"Epoch #{epoch+1} Training loss: {training_loss:.4f}")
        
        # VALIDATION PHASE PART
        writing_training_logs(f'{path_training_logs}', 
                              f"Start validation for Epoch #{epoch+1}...")
        print(f"Starting validation for Epoch #{epoch+1}...")

        val_loss = loss_epoch(
            process_name="Validation", 
            loss_tracker=validation_loss_tracker,
            data_loader=valid_data_loader,
            model=model,
            optimizer=None,
            device=device,
            path_to_save=path_to_save
        )
        loss_history["val"].append(val_loss)
        
        writing_training_logs(f'{path_training_logs}',
                              f"Epoch #{epoch+1} Validation loss: {val_loss:.4f}")
        
        if val_loss < best_validation_loss:
            best_validation_loss = val_loss 
            writing_training_logs(f'{path_training_logs}',
                                  f"New best validation loss: {best_validation_loss:.4f} at epoch #{epoch+1}. Saving model...")
            print(f"New best validation loss: {best_validation_loss:.4f} at epoch #{epoch+1}. Saving model...")
            
            # Save the model weights
            save_model(model, f"{path_best_model}")
            writing_training_logs(f'{path_training_logs}',
                                  f"Model weights saved to {path_best_model}")

        writing_training_logs(f'{path_training_logs}',
                                f"Finished Training.")
        print(f"Training logs can be found at {path_training_logs}")
        print(f"Best Weight Model can be found at {path_best_model}")
    return loss_history
