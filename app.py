import streamlit as st
import torch
import torchvision
from PIL import Image
import io

# Function to load Faster R-CNN model
def load_fasterrcnn_model(model_file):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    model.load_state_dict(torch.load(model_file))
    model.eval()
    return model

# Function to load RTMDet model
def load_rtmdet_model(model_file):
    model = torch.load(model_file)
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
    elif model_type == 'RTMDet':
        image_tensor = torch.tensor(image).unsqueeze(0)
        with torch.no_grad():
            prediction = model(image_tensor)
        return prediction

# Streamlit app layout
st.title('Object Detection with Multiple Models')

# Model selection dropdown
model_choice = st.selectbox('Select Model', ['Select a model', 'Faster R-CNN', 'RTMDet'])

# Upload an image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Upload model files
fasterrcnn_model_file = st.file_uploader("Upload Faster R-CNN model", type=["pth"])
rtmdet_model_file = st.file_uploader("Upload RTMDet model", type=["pth"])

# Initialize the model variable
model = None

# Load model button
if st.button('Load Model'):
    if model_choice == 'Faster R-CNN' and fasterrcnn_model_file is not None:
        model = load_fasterrcnn_model(fasterrcnn_model_file)
        st.success("Faster R-CNN model loaded successfully.")
    elif model_choice == 'RTMDet' and rtmdet_model_file is not None:
        model = load_rtmdet_model(rtmdet_model_file)
        st.success("RTMDet model loaded successfully.")
    else:
        st.error('Please select a valid model and upload the model file.')

# Show uploaded image
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Run inference if the model is loaded
    if model is not None:
        if st.button('Run Inference'):
            if model_choice == 'Faster R-CNN':
                prediction = perform_inference(model, image, model_choice)
                st.write("Prediction Results:")
                st.write(prediction)
            elif model_choice == 'RTMDet':
                prediction = perform_inference(model, image, model_choice)
                st.write("RTMDet Prediction Results:")
                st.write(prediction)
    else:
        st.error('Please load a model first.')
