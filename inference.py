import torch

import numpy as np
from matplotlib import pyplot as plt
import cv2

from model.classification.model import build_model_classification
from model.detection.model import build_model_object_detection
from model.utils import load_model
import argparse

# Argument parser for input file_name
parser = argparse.ArgumentParser(description="Run inference on an input image.")
parser.add_argument(
    "--file_name",
    type=str,
    required=True,
    help="Path to the input image file."
)
args = parser.parse_args()

# Use the file_name argument
file_name = args.file_name

# Check if the file exists
if not file_name:
    raise ValueError("No input file provided. Please specify a valid image file path.")
if not file_name.endswith(('.jpg', '.jpeg', '.png')):
    raise ValueError("Invalid file format. Please provide a valid image file (jpg, jpeg, png).")
if not cv2.os.path.exists(file_name):
    raise FileNotFoundError(f"The specified file does not exist: {file_name}")

class_names = {
    0: 'City Car',
    1: 'Big Truck',
    2: 'Multi Purpose Vehicle',
    3: 'Sedan',
    4: 'Sport Utility Vehicle',
    5: 'Truck',
    6: 'Van'
}

# Load the models
model_detection = build_model_object_detection(backbone='resnet50', num_class=2, use_pretrained=False)
model_classification = build_model_classification(class_num=7, use_pretrained=False)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
cpu_device = torch.device("cpu")

load_model(model_detection, 'logs/detection/best_model/best_model_weights.pth', device)
load_model(model_classification, 'logs/classification/best_model/best_model_weights.pth', device)

model_detection = model_detection.to(device)
model_classification = model_classification.to(device)
model_detection.eval()
model_classification.eval()

# Read the Input Image
# file_name = f"LabelChangeTest-1/test/66_jpg.rf.917134f4a858497596e301a2868bcdbe.jpg" 

image = cv2.imread(f'{file_name}', cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
image /= 255.0

image_preprocess = torch.from_numpy(image).permute(2,0,1).to(device).unsqueeze(0)

outputs = model_detection(image_preprocess)
outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

boxes = outputs[0]['boxes']
scores = outputs[0]['scores']

confidence_threshold = 0.5 # Adjust as needed
filtered_boxes = boxes[scores > confidence_threshold].detach().cpu().numpy().astype(np.int32)

img_predictions = []

# Iterate for each box prediction 
for bbox in filtered_boxes:
    if bbox.ndim != 1 or bbox.shape[0] != 4:
        print(f"Warning: Unexpected bbox format for image {bbox}. Skipping.")
        img_predictions.append(-2) # Indicate skipped due to format
        continue
    x_min, y_min, x_max, y_max = [int(b.item()) for b in bbox]

    height, width = image_preprocess[0].shape[-2:] # If the format is cxhxw

    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(width, x_max)
    y_max = min(height, y_max)

    classifier_device = model_classification.fc.weight.device

    roi_np = image_preprocess[0][:, y_min:y_max, x_min:x_max].unsqueeze(0)
    roi_np = roi_np.to(classifier_device)

    with torch.no_grad(): # No need to calculate gradients for inference
        output = model_classification(roi_np)
    probabilities = torch.softmax(output, dim=1)

    _, predicted_class = torch.max(probabilities, 1)
    img_predictions.append(predicted_class.item())

# Import the image for visualization
img = image.copy()
fig, ax = plt.subplots(1, 1, figsize=(16, 8))
labels=img_predictions

for i, box in enumerate(filtered_boxes):
    x_min, y_min, x_max, y_max = box.astype(np.int32)
    # score = scores[i]
    label_index = labels[i]
    label = class_names.get(label_index, f'Class {label_index}') # Get class name or use index

    # Draw the bounding box
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (220, 0, 0), 3)

    # Put the label and score text
    text = f'{label}'
    # Determine text location to avoid overlapping with the box
    text_x = x_min
    text_y = y_min - 10 if y_min - 10 > 10 else y_min + 10
    cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


ax.set_axis_off()
ax.imshow(img)
plt.show()

# Save the plot to a file
output_file = 'output_image_with_boxes.png'
plt.savefig(output_file, bbox_inches='tight', dpi=300)
print(f"Plot saved to {output_file}")