
# Commented out IPython magic to ensure Python compatibility.
# %%capture
# # Install Roboflow
# !pip install roboflow
# 
# # Install Ultralytics (YOLOv8)
# !pip install ultralytics
# 
# # Install YOLO dependencies
# !pip install torch torchvision torchaudio
# !pip install matplotlib

# data from roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="c0AKlRxmACmG83mIunVG")
project = rf.workspace("su2-d7z1e").project("road-accident-victim-detection-bytzk")
version = project.version(2)
dataset = version.download("yolov8")

#"""Zkouska ruznych parametru"""

from ultralytics import YOLO
import torch
import os

# Get the number of available GPUs, else use CPU
ngpus = torch.cuda.device_count() if torch.cuda.is_available() else 'cpu'

# Define the path to your dataset
DATA_PATH = f"{dataset.location}/data.yaml"  # Automatically set by Roboflow

# Define a list of weight configurations to experiment with
weight_configs = [
    #{'box': 0.05, 'cls': 0.5, 'obj': 1.0, 'name': 'Default_Weights'},
    {'box': 0.1, 'cls': 0.5, 'obj': 1.0, 'name': 'Increased_Box_Loss'},
    #{'box': 0.05, 'cls': 0.6, 'obj': 1.0, 'name': 'Increased_Cls_Loss'},
   #{'box': 0.05, 'cls': 0.5, 'obj': 1.2, 'name': 'Increased_Obj_Loss'},
]

# Results directory
results_dir = "accident_detection_results"
os.makedirs(results_dir, exist_ok=True)

# Train models with different weight configurations
for config in weight_configs:
    print(f"Training model with configuration: {config['name']}")

    # Initialize YOLO model
    model = YOLO('yolov8s.pt')  # Use a smaller variant for faster computation

    # Train the model with the current weight configuration
    model.train(
        data=DATA_PATH,           # Path to data.yaml
        epochs=10,                # Number of training epochs
        imgsz=640,                # Image size
        batch=16,                 # Batch size
        device=ngpus,             # Use GPU or CPU
        project=results_dir,      # Project directory
        name=config['name'],      # Experiment name
        exist_ok=True,            # Overwrite if run exists
        box=config['box'],        # Weight for bounding box loss
        cls=config['cls'],        # Weight for classification loss
        kobj=config['obj'],        # Weight for objectness loss
    )

    # Save the best model for this configuration
    best_model_path = f"{results_dir}/{config['name']}_best.pt"
    # Save the best model
    model.export(format='onnx')  # Exporting as ONNX for flexibility
    model.save(best_model_path)
    print(f"Best model for {config['name']} saved at {best_model_path}\n")

# Summarize and analyze results (Optional)
print("Training complete. Models and results saved in:", results_dir)