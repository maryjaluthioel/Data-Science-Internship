import os
from ultralytics import YOLO

def train_yolo_model():
    print("--- Initializing Supervised Learning Training Pipeline ---")
    
    # 1. Load a pre-trained baseline nano network architecture
    model = YOLO("yolov8n.pt")
    
    # 2. Mock configuration setup for a dataset yaml pointer
    # Replace 'coco8.yaml' with your custom Kaggle dataset config file path (e.g., 'data.yaml')
    dataset_yaml = "coco8.yaml" 
    
    print(f"Loading dataset specification configuration: {dataset_yaml}")
    
    # 3. Execute model training (Supervised Learning Regression + Classification Optimization)
    # Adjust epochs, batch sizes, and hardware parameters depending on setup requirements
    results = model.train(
        data=dataset_yaml,
        epochs=10,          # Set higher (e.g., 50-100) for convergence on custom Kaggle data
        imgsz=640,          # Canonical preprocessed image resolution tensor size
        batch=4,            # Batched processing index boundaries 
        device="cpu",       # Change to '0' or 'cuda' if local GPU execution hardware is ready
        workers=2,
        project="AI_Surveillance_Model",
        name="YOLOv8_Custom_Run"
    )
    
    print("\n--- Model Evaluation Complete ---")
    print("Exporting optimized parameters to structural file locations...")
    
    # 4. Force save standard serialization weights format
    metrics = model.val()
    print(f"Mean Average Precision (mAP@50): {metrics.results_dict['metrics/mAP50(B)']:.4f}")
    print("Training job finalized successfully. Run 'app.py' to launch display nodes.")

if __name__ == "__main__":
    train_yolo_model()