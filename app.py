import streamlit as st
from roboflow import Roboflow
from PIL import Image
import torch

# Initialize Roboflow with your API key
rf = Roboflow(api_key="RFaNdGHxTtn46bvxSFvM")

# Access the project and version from your workspace
project = rf.workspace("yolo-wood").project("project-design-ekhku")
version = project.version(2)

# Download the YOLOv5 model from Roboflow (with YOLOv5 format)
dataset = version.download("yolov5")

# Load the YOLOv5 model directly from the dataset (Roboflow API handles the model)
model = torch.hub.load('ultralytics/yolov5', 'custom', path=dataset.location + '/yolov5.pt')

# Streamlit UI
st.title('YOLOv5 Object Detection')

# Upload image from the UI
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image using PIL
    img = Image.open(uploaded_file)
    
    # Display the uploaded image
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # Run inference on the uploaded image
    results = model(img)  # The model makes a prediction on the image
    
    # Display detection results
    st.subheader("Detection Results")
    results.show()  # This will show the image with bounding boxes
    
    # Show the predictions in a tabular format (labels and confidence)
    st.write("Predictions:", results.pandas().xywh)
