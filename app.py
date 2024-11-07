import streamlit as st
import torch
import torchvision
from PIL import Image
import requests
from io import BytesIO
from pathlib import Path

# Function to load Faster R-CNN model
def load_fasterrcnn_model(model_path):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Function to load YOLOv5 model via Roboflow API
def load_yolov5_model(api_key, project_name, version):
    from roboflow import Roboflow
    rf = Roboflow(api_key=api_key)
    model = rf.workspace().project(project_name).version(version).model
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

# Streamlit app layout
st.title('Soft Des 3 Model designs')

# Model selection dropdown
model_choice = st.selectbox('Select Model', ['Select a model', 'Faster R-CNN', 'YOLOv5', 'RTMDet'])

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Load model button
if st.button('Load Model'):
    if model_choice == 'Faster R-CNN':
        model_path = st.text_input("Enter the path to the Faster R-CNN model (.pth):")
        if model_path:
            model = load_fasterrcnn_model(model_path)
            st.success("Faster R-CNN model loaded successfully.")
    elif model_choice == 'YOLOv5':
        api_key = st.text_input("Enter your Roboflow API key:")
        project_name = st.text_input("Enter your project name:")
        version = st.number_input("Enter the version number of the YOLOv5 model:", min_value=1)
        if api_key and project_name and version:
            model = load_yolov5_model(api_key, project_name, version)
            st.success("YOLOv5 model loaded successfully.")
    elif model_choice == 'RTMDet':
        model_path = st.text_input("Enter the path to the RTMDet model:")
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

