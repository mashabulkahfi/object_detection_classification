import torch 
from matplotlib import pyplot as plt

def save_model(model, path_to_save):
    """
    Save the model weights to the specified path.
    
    Args:
        model: The trained model.
        path_to_save: Path to save the model weights.
    """
    torch.save(model.state_dict(), path_to_save)
    print(f"Model weights saved to {path_to_save}")

def load_model(model, path_load_from, device):
    """
    Save the model weights to the specified path.
    
    Args:
        model: The trained model.
        path_to_save: Path to save the model weights.
    """
    # path_load_from = '/content/best_model_weights.pth'
    model.load_state_dict(torch.load(path_load_from, map_location=device))
    print(f"Model weights loaded successfully from {path_load_from}")

def writing_training_logs(path_to_save, message):
    """
    Write training logs to a file.
    
    Args:
        path_to_save: Path to save the training logs.
        message: Message to write in the log file.
    """
    with open(f'{path_to_save}', 'a') as f:
        f.write(message + '\n')
    print(message)

def plot_loss_history(loss_history, path_to_save):
    """
    Plot the training and validation loss history.
    
    Args:
        loss_history: Dictionary containing training and validation loss history.
        path_to_save: Path to save the plot.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history['train'], label='Training Loss')
    plt.plot(loss_history['val'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss History')
    plt.legend()
    plt.grid()
    plt.savefig(f'{path_to_save}/loss_history.png')
    plt.show()

def plot_metric_history(metric_history, path_to_save):
    """
    Plot the training and validation metric history.
    
    Args:
        metric_history: Dictionary containing training and validation metric history.
        path_to_save: Path to save the plot.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(metric_history['train'], label='Training Metric')
    plt.plot(metric_history['val'], label='Validation Metric')
    plt.xlabel('Epochs')
    plt.ylabel('Metric')
    plt.title('Training and Validation Metric History')
    plt.legend()
    plt.grid()
    plt.savefig(f'{path_to_save}/metric_history.png')
    plt.show()
