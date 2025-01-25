import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os
import pytesseract
from pytesseract import Output

# Streamlit App Configuration
st.set_page_config(page_title='YOLO Object Detection', page_icon="🚗")
st.title('YOLO Image and Video Processing')
st.subheader('License Plate Object Detection & Text Extraction')

# File Uploader
uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov", "mkv"])

# Load YOLO Model
MODEL_PATH = "D:\\Object-Detection\\Streamlit-Application\\best.pt"
model = YOLO(MODEL_PATH)
# st.success("Model loaded successfully!")

# Function to perform OCR

def extract_text_from_image(image, x1, y1, x2, y2):
    """
    Extract text from a specified region of an image using pytesseract.

    Parameters:
    image (np.array): The input image.
    x1, y1, x2, y2 (int): Coordinates of the bounding box.

    Returns:
    str: Extracted text.
    """
    # Specify Tesseract executable path
    pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
    roi = image[y1:y2, x1:x2]
    # gray_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    text = pytesseract.image_to_string(roi, config='--psm 6')
    return text

# Function to Process Images
def predict_and_save_image(image_path, output_image_path):
    """
    Detect objects in an image, draw bounding boxes, extract text, and save the result.

    Parameters:
    image_path (str): Path to the input image.
    output_image_path (str): Path to save the processed image.

    Returns:
    Tuple: Output image path and detected text list.
    """
    results = model.predict(image_path, device='cpu')
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detected_texts = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f'{confidence*100:.2f}%', (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            text = extract_text_from_image(image, x1, y1, x2, y2)
            detected_texts.append(text)
            st.write(f"Detected Text: {text}")

    # Save the processed image
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_image_path, image)
    return output_image_path, detected_texts

# Function to Process Videos
def predict_and_plot_video(video_path, output_path):
    """
    Predicts and saves the bounding boxes on the given test video using the trained YOLO model.

    Parameters:
    video_path (str): Path to the test video file.
    output_path (str): Path to save the output video file.

    Returns:
    str: The path to the saved output video file.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error(f"Error opening video file: {video_path}")
            return None
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model.predict(rgb_frame, device='cpu')
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = box.conf[0]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'{confidence*100:.2f}%', (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            out.write(frame)
        cap.release()
        out.release()
        return output_path
    except Exception as e:
        st.error(f"Error processing video: {e}")
        return None
# Main Media Processing Workflow
def process_media(input_path, output_path):
    """
    Processes the uploaded media file (image or video) and returns the path to the output file.

    Parameters:
    input_path (str): Path to the input media file.
    output_path (str): Path to save the output media file.

    Returns:
    str: Path to the processed media file.
    """
    file_extension = os.path.splitext(input_path)[1].lower()
    if file_extension in ['.mp4', '.avi', '.mov', '.mkv']:
        return predict_and_plot_video(input_path, output_path)
    elif file_extension in ['.jpg', '.jpeg', '.png', '.bmp']:
        return predict_and_save_image(input_path, output_path)
    else:
        st.error(f"Unsupported file type: {file_extension}")
        return None

# Handle Uploaded File
if uploaded_file is not None:
    input_path = f"temp/{uploaded_file.name}"
    output_path = f"output/{uploaded_file.name}"

    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.write("Processing...")

    if input_path.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        result_path = process_media(input_path, output_path)
        if result_path:
            video_file = open(result_path, 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)
            video_file.close()
    else:
        result_path, detected_texts = process_media(input_path, output_path)
        if result_path:
            st.image(result_path)
            # if detected_texts:
            #     st.write("Detected Texts:")
            #     for text in detected_texts:
            #         st.write(f"- {text}")
