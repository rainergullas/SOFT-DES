import streamlit as st
import torch
import torchvision
from PIL import Image
from roboflow import Roboflow
import requests
from io import BytesIO
from pathlib import Path
from google.colab import drive

# Function to load Faster R-CNN model
def load_fasterrcnn_model(model_path):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Function to load YOLOv5 model via Roboflow API
def load_yolov5_model():
    # Directly using the given API key and project details
    rf = Roboflow(api_key="RFaNdGHxTtn46bvxSFvM")
    project = rf.workspace("yolo-wood").project("project-design-ekhku")
    version = project.version(2)
    dataset = version.download("yolov5")
    model = dataset.model
    return model

# Function to load RTMDet model
def load_rtmdet_model(model_path):
    model = torch.load(model_path)
    model.eval()
    return model

# Function for performing inference on the image
def perform_inference(model, image, model_type):
    if model_type == 'Faster R-CNN':
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
        image_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            prediction = model(image_tensor)
        return prediction
    elif model_type == 'YOLOv5':
        # Perform inference using YOLOv5
        results = model.predict(image)
        return results
    elif model_type == 'RTMDet':
        # Perform inference using RTMDet
        image_tensor = torch.tensor(image).unsqueeze(0)
        with torch.no_grad():
            prediction = model(image_tensor)
        return prediction

# Function to mount Google Drive
def mount_drive():
    drive.mount('/content/drive')
    st.success("Google Drive mounted successfully!")

# Streamlit app layout
st.title('Soft Des Three Model Demonstration')

# Model selection dropdown
model_choice = st.selectbox('Select Model', ['Select a model', 'Faster R-CNN', 'YOLOv5', 'RTMDet'])

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Load model button
if st.button('Load Model'):
    if model_choice == 'Faster R-CNN':
        mount_drive()
        model_path = st.text_input("/content/drive/MyDrive/Copy of model_final.pth")
        if model_path:
            model = load_fasterrcnn_model(model_path)
            st.success("Faster R-CNN model loaded successfully.")
    elif model_choice == 'YOLOv5':
        model = load_yolov5_model()
        st.success("YOLOv5 model loaded successfully.")
    elif model_choice == 'RTMDet':
        mount_drive()
        model_path = st.text_input("/content/drive/MyDrive/epoch_2.pth")
        if model_path:
            model = load_rtmdet_model(model_path)
            st.success("RTMDet model loaded successfully.")
    else:
        st.error('Please select a valid model first.')

# Show uploaded image
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Run inference
    if st.button('Run Inference'):
        if model_choice != 'Select a model':
            if model_choice == 'Faster R-CNN':
                prediction = perform_inference(model, image, model_choice)
                st.write("Prediction Results:")
                st.write(prediction)
            elif model_choice == 'YOLOv5':
                results = perform_inference(model, image, model_choice)
                st.write("YOLOv5 Prediction Results:")
                st.write(results)
            elif model_choice == 'RTMDet':
                prediction = perform_inference(model, image, model_choice)
                st.write("RTMDet Prediction Results:")
                st.write(prediction)
        else:
            st.error('Please load a model first.')
