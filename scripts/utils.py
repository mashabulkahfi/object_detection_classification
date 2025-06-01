import os 
import pandas as pd
import shutil

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

def assign_set_refactored(group, ratio=0.8, first_group='Training', second_group='Test'):
    """Assigns a set group to each item in the group based on the specified ratio.
    Args:
        group (pd.DataFrame): A DataFrame group containing items to be assigned to sets.
        ratio (float): The ratio of the first group size to the total size of the group. Default is 0.8.
        first_group (str): The name of the first group to assign. Default is 'Training'.
        second_group (str): The name of the second group to assign. Default is 'Test'.
    Returns:
        pd.Series: A Series with the same index as the group, containing the assigned set group names.
    """
    split_index = int(len(group) * ratio)  # 80% for training
    # Create a Series to hold the set assignments for this group
    set_assignments = pd.Series(index=group.index, dtype='object')
    set_assignments.iloc[:split_index] = first_group
    set_assignments.iloc[split_index:] = second_group

    return set_assignments

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

def copy_images_to_split_dirs(df, dataset_type, original_base_path, new_base_path):
    """
    Copies images from original dataset path to new split directories based on DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing image information and split group.
        dataset_type (str): The type of dataset ('train', 'val', 'test').
        original_base_path (str): The base directory of the original dataset.
        new_base_path (str): The base directory where the new split dataset will be created.
    """
    print(f"Copying {dataset_type} images...")
    count = 0
    for index, row in df.iterrows():
        class_name = row['class_name']
        file_name = row['file_name']

        # Original image path
        original_image_path = os.path.join(original_base_path, class_name, file_name)

        # New image path
        new_dataset_type_dir = os.path.join(new_base_path, dataset_type)
        new_class_dir = os.path.join(new_dataset_type_dir, class_name)
        new_image_path = os.path.join(new_class_dir, file_name)

        # Create target class directory if it doesn't exist
        os.makedirs(new_class_dir, exist_ok=True)

        # Check if original file exists before copying
        if os.path.exists(original_image_path):
            # Copy the image file
            try:
                shutil.copy2(original_image_path, new_image_path)
                count += 1
            except Exception as e:
                print(f"Error copying {original_image_path} to {new_image_path}: {e}")
        else:
            print(f"Warning: Original file not found: {original_image_path}")

    print(f"Finished copying {count} {dataset_type} images.")

    
def move_dataset_detection(source_folder, destination_folder, remove=False):
    """
    Moves specified folders from the source directory to the destination directory.
    Args:
        source_folder (str): The path to the source directory containing the dataset.
        destination_folder (str): The path to the destination directory where folders will be copied.
        remove (bool): If True, removes the source directory after copying. Default is False.
    """
    source_dir = source_folder
    destination_dir = destination_folder

    # Define the folders to copy
    folders_to_copy = ['test', 'train', 'valid']

    print(f"Contents of the downloaded directory: {source_dir}")
    try:
        print(os.listdir(source_dir))
    except FileNotFoundError:
        print(f"Error: The source directory {source_dir} was not found.")


    # Copy each folder
    for folder in folders_to_copy:
        source_folder_path = os.path.join(source_dir, folder)
        destination_folder_path = os.path.join(destination_dir, folder)
        print(f"Copying {source_folder_path} to {destination_folder_path}")
        try:
            shutil.copytree(source_folder_path, destination_folder_path, dirs_exist_ok=True)
        except FileNotFoundError:
            print(f"Error: Source folder {source_folder_path} not found. Please check the directory structure of the downloaded dataset.")


    print("Finished copying specified folders.")

    # Delete the source folder
    # Remove the source directory and its contents
    
    if remove:
        if os.path.exists(source_dir):
            print(f"Removing source directory: {source_dir}")
            shutil.rmtree(source_dir)
            print(f"Source directory {source_dir} removed successfully.")
        else:
            print(f"Source directory {source_dir} does not exist. No action needed.")
