import os
from .utils import list_files_by_folder, preprocess_label, assign_set_refactored, copy_images_to_split_dirs
from pathlib import Path

current_file_path = Path(__file__)
project_dir = current_file_path.parent.parent
source_dir = f'{project_dir}/data/classification/raw/vehicle-images-dataset'

dataset_df = list_files_by_folder(source_dir)
dataset_df['Set'] = None  # Initialize the 'Set' column

# Shuffle the DataFrame to randomize the order
dataset_df = dataset_df.sample(frac=1, random_state=42).reset_index(drop=True)

dataset_df['label_class'] = dataset_df['Folder'].apply(lambda cat: preprocess_label(cat))
dataset_df['Set'] = dataset_df.groupby('Folder', group_keys=False).apply(assign_set_refactored)

dataset_df = dataset_df.reset_index()
dataset_df.columns=['image_id','class_name', 'file_name', 'set_group','label_class']

# Split the dataset into training and test sets
df_train = dataset_df[dataset_df['set_group']=="Training"]
df_test_all = dataset_df[dataset_df['set_group']=="Test"]

# Assign the set group for the test set, ensuring a 50/50 split between Test and Validation
df_test_all['set_group'] = df_test_all.groupby('class_name', group_keys=False).apply(assign_set_refactored, ratio=0.5, first_group='Test', second_group='Validation')

# Split the dataset into training and test sets
df_val = df_test_all[df_test_all['set_group']=="Validation"]
df_test = df_test_all[df_test_all['set_group']=="Test"]

# Define the base directory for the new split dataset
destination_dir = f'{project_dir}/data/classification'

# Ensure the base directory for the new split dataset exists
os.makedirs(destination_dir, exist_ok=True)

# Copy images for each split
copy_images_to_split_dirs(df_train, 'train', source_dir, destination_dir)
copy_images_to_split_dirs(df_val, 'val', source_dir, destination_dir)
copy_images_to_split_dirs(df_test, 'test', source_dir, destination_dir)

print("Dataset for Image Classification has been successfully created and images copied to respective directories.")