from .utils import move_dataset_detection
from pathlib import Path
from roboflow import Roboflow
import argparse

current_file_path = Path(__file__)
project_dir = current_file_path.parent.parent

parser = argparse.ArgumentParser(description="Run Train Classification model.")
parser.add_argument(
    "--api_roboflow_key",
    type=str,
    required=True,
    help="API key for Roboflow (required)"
)
args = parser.parse_args()
roboflow_api_key = args.api_roboflow_key


# Replace with your actual API key
# rf = Roboflow(api_key="1WzSCPbUIO9kWfx9keuC")
rf = Roboflow(api_key=roboflow_api_key)

# Replace with your workspace and project names
project = rf.workspace("dallmeier").project("labelchangetest")

# Replace with the desired version number
version = project.version(1)

# Define the directory where you want to download the dataset
# download_dir = f"{root_dir}/data/detection/" # You can change this to your desired path
dataset = version.download("coco")



source_dir = f"{project_dir}/LabelChangeTest-1"
destination_dir = f"{project_dir}/data/detection"
move_dataset_detection(source_dir, destination_dir, remove=True)

print(f"Dataset downloaded to: {destination_dir}")