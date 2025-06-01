import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import pandas as pd
from ..utils import save_model, writing_training_logs


class CustomImageCarDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.image_dir = image_dir
        self.df = df
        self.image_ids = list(self.df['image_id'])
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        records = self.df[self.df['image_id'] == image_id]

        image_path = f"{records['file_path'].values[0]}"
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        original_height, original_width = image.shape[:2]
        if original_height > 64 or original_width > 64:
            interpolation = cv2.INTER_AREA  # Downsampling
        else:
            interpolation = cv2.INTER_CUBIC  # Upsampling
        resized_image = cv2.resize(image, (64, 64), interpolation=interpolation)


        processed_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB).astype(np.float32)
        processed_image /= 255.0

        #Transpose the image tensor to be (channels, height, width)
        #from (height, width, channels)
        processed_image = processed_image.transpose((2, 0, 1))

        #Convert numpy array to torch tensor
        processed_image = torch.from_numpy(processed_image)

        labels = records['label_class'].values
        labels = torch.tensor(labels[0], dtype=torch.long)

        if self.transform:
            processed_image = self.transform(processed_image)

        return processed_image, labels
    
# Function to list files in each folder and return a DataFrame
def list_files_by_folder(base_path):
    """
    Lists all files in each subfolder of the given base path and returns a DataFrame.
    Args:
        base_path (str): The base directory containing subfolders with files.
    Returns:
        pd.DataFrame: A DataFrame with columns 'Folder' and 'File', where 'Folder' is the name of the subfolder
        and 'File' is the name of the file within that subfolder."""
    data = []

    # Iterate through each folder in the base path
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)

        # Check if it's a directory
        if os.path.isdir(folder_path):
            # List all files in the folder
            files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
            for file in files:
                data.append({"Folder": folder_name, "File": file})

    return pd.DataFrame(data)

def preprocess_label(cat):
    """
    Preprocesses the label category to a numerical format.
    Args:
        cat (str): The category label as a string.
    Returns:
        int: A numerical representation of the category.
    """
    match cat:
        case "City Car":
             return 0
        case "Big Truck":
             return 1
        case "Multi Purpose Vehicle":
             return 2
        case "Sedan":
            return 3
        case "Sport Utility Vehicle":
            return 4
        case "Truck":
            return 5
        case "Van":
            return 6
        
# Also define the metric_batch function, assuming it calculates accuracy per batch
def metric_batch(output, target, check_id):
    """
    Calculates the accuracy for a batch of predictions.
    Args:
        output (torch.Tensor): The model's output predictions.
        target (torch.Tensor): The ground truth labels.
        check_id: Not used in this function, but included for compatibility.
    Returns:
        float: The accuracy of the predictions for the batch.
    """
    # calculate accuracy
    _, predicted = torch.max(output.data, 1)

    target_squeezed = target.squeeze()
    if predicted.shape != target_squeezed.shape:
        pass

    correct = (predicted == target.squeeze()).sum().item()
    metric_b = correct / target.size(0)
    return metric_b


def loss_epoch(model, loss_func, data_loader, sanity_check=False, opt=None, device=None):
    """
    Computes the loss and metric for an entire epoch.
    Args:
        model: The model to evaluate.
        loss_func: The loss function to use.
        data_loader: DataLoader for the dataset.
        sanity_check (bool): If True, only processes one batch for quick checks.
        opt: Optimizer for training phase (if None, only evaluation is performed).
        device: Device to run the computations on (e.g., 'cuda' or 'cpu').
    Returns:
        tuple: Average loss and metric for the epoch.
    """
    running_loss = 0.0
    running_metric = 0.0
    len_data = len(data_loader.dataset)

    for i, (inputs, labels) in enumerate(data_loader):
        # Move inputs and labels to the specified device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # If in training phase, zero the gradients
        if opt is not None:
            opt.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Calculate the loss for the batch
        loss = loss_func(outputs, labels)

        # Calculate the metric for the batch
        metric_b = metric_batch(outputs, labels, None)

        # Backward pass and optimizer step (if training)
        if opt is not None:
            loss.backward()
            opt.step()

        # Accumulate the running loss and metric
        running_loss += loss.item() * inputs.size(0)  # Multiply by batch size
        running_metric += metric_b * inputs.size(0)  # Multiply by batch size

        # If sanity check is enabled, break after one batch
        if sanity_check:
            break

    # Calculate the average loss and metric for the epoch
    loss = running_loss / len_data
    metric = running_metric / len_data

    return loss, metric
    
def load_dataset(base_path):
    """Loads the dataset from the specified base path and returns DataFrames for train, validation, and test sets."""
    
    df_train = process_df(base_path, 'train')
    df_val = process_df(base_path, 'val')
    # df_test = process_df(base_path, 'test')

    return df_train, df_val

def process_df(base_path, type_dataset):
    """Processes the DataFrame to include the full file path for images."""

    
    df = list_files_by_folder(f"{base_path}/{type_dataset}")
    df['label_class'] = df['Folder'].apply(lambda cat: preprocess_label(cat))
    df = df.reset_index()
    df.columns = ['image_id', 'class_name', 'file_name', 'label_class']
    df['file_path'] = df.apply(lambda row: f"{base_path}/{type_dataset}/{row['class_name']}/{row['file_name']}", axis=1)

    return df 

def training_model(train_dl,
                   valid_dl,
                   model,
                   optimiser,
                   device,
                   num_epochs=2,
                   lr_scheduler=None,
                   loss_func=None,
                   path_to_save=None):
    """
    Trains the model for a specified number of epochs and evaluates it on the validation dataset.
    Args:
        train_dl: DataLoader for the training dataset.
        valid_dl: DataLoader for the validation dataset.
        model: The model to train.
        optimiser: Optimizer for training.
        device: Device to run the computations on (e.g., 'cuda' or 'cpu').
        num_epochs (int): Number of epochs to train the model.
        lr_scheduler: Learning rate scheduler (optional).
        loss_func: Loss function to use for training.
        path_to_save: Path to save the best model weights and training logs.
    Returns:
        tuple: History of loss and metric values for training and validation datasets.
    """
    loss_history={"train": [],"val": []} # history of loss values in each epoch
    metric_history={"train": [],"val": []}
    best_loss=float('inf')

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        model.train()  # Set the model to training mode
        # Path to save : root_dir + /logs/classification
        writing_training_logs(f'{path_to_save}/training_history/training_logs.txt', 
                                f"Starting Epoch #{epoch+1}...")
        print(f"Starting training for Epoch #{epoch+1}...")
        train_loss, train_metric = loss_epoch(model, 
                                              loss_func, 
                                              train_dl, 
                                              sanity_check=False, 
                                              opt=optimiser, 
                                              device=device)

        loss_history["train"].append(train_loss)
        metric_history["train"].append(train_metric)

        # learning rate schedule
        if lr_scheduler is not None:
            lr_scheduler.step()

        print(f"Epoch #{epoch+1} Training loss: {train_loss}, Training metric: {train_metric}")
        writing_training_logs(f'{path_to_save}/training_history/training_logs.txt', 
                              f"Epoch #{epoch+1} Training loss: {train_loss:.4f}, Training metric: {train_metric}")

        # evaluate model on validation dataset

        writing_training_logs(f'{path_to_save}/training_history/training_logs.txt', 
                              f"Start validation for Epoch #{epoch+1}...")
        print(f"Starting validation for Epoch #{epoch+1}...")

        model.eval()
        with torch.no_grad():
            val_loss, val_metric=loss_epoch(model,
                                            loss_func,
                                            valid_dl, 
                                            sanity_check=False, 
                                            opt=None, 
                                            device=device)
        
        # collect loss and metric for validation dataset
        loss_history["val"].append(val_loss)
        metric_history["val"].append(val_metric)
        print(f"Epoch #{epoch+1} Validation loss: {val_loss}, Validation metric: {val_metric}")
        writing_training_logs(f'{path_to_save}/training_history/training_logs.txt', 
                              f"Epoch #{epoch+1} Validation loss: {val_loss:.4f}, Validation metric: {val_metric:.4f}")
        # store best model
        if val_loss < best_loss:
            best_loss = val_loss
            
            writing_training_logs(f'{path_to_save}/training_history/training_logs.txt',
                                  f"New best validation loss: {best_loss:.4f} at epoch #{epoch+1}. Saving model...")
            print(f"New best validation loss: {best_loss:.4f} at epoch #{epoch+1}. Saving model...")

            save_model(model, f"{path_to_save}/best_model/best_model_weights.pth")
            # torch.save(model.state_dict(), path)
            # if(verbose):
            print("Copied best model weights!")

        writing_training_logs(f'{path_to_save}/training_history/training_logs.txt',
                              f"train loss: {train_loss:.6f}, val loss: {val_loss:.6f}, accuracy validation: {100*val_metric:.2f}")

    writing_training_logs(f"{path_to_save}/training_history/training_logs.txt", 'Finished Training')
    return loss_history, metric_history
