import streamlit as st
import cv2
import torch
from pathlib import Path
import tempfile
import sys
import numpy as np

# Import YOLOv5 model and utilities
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device

# Load the custom YOLOv5 model
model_path = 'runs/train/ball_person_model2/weights/best.pt'
device = select_device('')  # Use CUDA if available
model = DetectMultiBackend(model_path, device=device, dnn=False)
img_size = 640  # Set input size to 640x640 for model

# CSS for styling with a customized look
st.markdown("""
    <style>
        /* Expand width of main container */
        .main {
            max-width: 1050px;
            padding: 22px;
            background-color: #e7eff6;
            border-radius: 17px;
            box-shadow: 0px 6px 14px rgba(30, 30, 30, 0.15);
        }

        /* Full-screen container with a customized gradient */
        .css-18e3th9 {
            background: linear-gradient(128deg, #d2ef9d, #94cda1, #a4e6f3);
        }

        /* Title styling with shadow and specific font */
        h1 {
            color: #364162;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            text-shadow: 1px 2px 6px rgba(0,0,0,0.3);
            letter-spacing: 1.2px;
        }

        /* Button styling with unique color and font */
        .stButton>button {
            background-color: #47b759;
            color: white;
            padding: 12px 22px;
            border-radius: 12px;
            border: 2px solid #1e7e34;
            cursor: pointer;
            font-weight: bold;
            font-size: 16px;
            font-family: Arial, sans-serif;
            text-shadow: 0px 1px 1px rgba(0, 0, 0, 0.2);
        }
        .stButton>button:hover {
            background-color: #3f9b50;
        }

        /* Progress bar styling */
        .stProgress .st-bs {
            background-color: #364162 !important;
            height: 18px;
            border-radius: 5px;
            box-shadow: 0px 3px 8px rgba(30, 30, 30, 0.15);
        }
    </style>
""", unsafe_allow_html=True)

st.title("Player and Ball Detection Application 🏓🏓🎾”)
st.write("Upload a video to detect players and balls using the YOLOv5 model.")

# File uploader for video input
uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])

if uploaded_video:
    # Create a temporary file to store the uploaded video
    temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_video_path.write(uploaded_video.read())
    temp_video_path.close()

    # Process button
    if st.button("Process"):
        # Load video and initialize parameters
        cap = cv2.VideoCapture(temp_video_path.name)
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        out = cv2.VideoWriter(output_path, fourcc, fps, (img_size, img_size))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        st.write("Processing video...")

        # Progress bar
        progress_bar = st.progress(0)

        # Process each frame
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame to 640x640 for YOLO input and output
            frame_resized = cv2.resize(frame, (img_size, img_size))

            # Prepare the frame for model input
            img = torch.from_numpy(frame_resized).to(device)
            img = img.permute(2, 0, 1).float() / 255.0  # Normalize and permute
            img = img.unsqueeze(0)  # Add batch dimension

            # Inference
            pred = model(img, augment=False, visualize=False)
            pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)

            # Process detections
            for det in pred:
                if len(det):
                    # Scale detections to 640x640 output size
                    det[:, :4] = scale_boxes((img_size, img_size), det[:, :4], (img_size, img_size)).round()

                    # Draw bounding boxes on the resized frame
                    for *xyxy, conf, cls in reversed(det):
                        x1, y1, x2, y2 = map(int, xyxy)
                        label = f'{model.names[int(cls)]} {conf:.2f}'
                        color = (0, 255, 0) if model.names[int(cls)] in ['player1', 'player2','person1','person2'] else (255, 0, 0)
                        cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color, 2)
                        if label:
                            t_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=1)[0]
                            cv2.rectangle(frame_resized, (x1, y1 - t_size[1] - 4), (x1 + t_size[0], y1), color, -1)  # Background for text
                            cv2.putText(frame_resized, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness=1)

            # Write processed frame to output
            out.write(frame_resized)

            # Update progress
            progress_percentage = int((i + 1) / total_frames * 100)
            progress_bar.progress(progress_percentage)

        # Release resources
        cap.release()
        out.release()

        st.success("Detection complete! ")

        # Display processed video
        st.video(output_path)

        # Provide download button
        with open(output_path, "rb") as file:
            st.download_button(
                label="Download Processed Video",
                data=file,
                file_name="processed_video.mp4",
                mime="video/mp4"
            )