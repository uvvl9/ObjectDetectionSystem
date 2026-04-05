from ultralytics import YOLO

# Start with a pre-trained model
model = YOLO('yolov8n.pt')

# Train the model on your custom dataset
# Replace 'path/to/your/data.yaml' with the actual path to your data.yaml file from Roboflow
results = model.train(data='C:/Users/abudi/Downloads/Drug_Detection.v4i.yolov11/data.yaml', epochs=50)

# Save the trained model
model.save('DrugsModel.pt')