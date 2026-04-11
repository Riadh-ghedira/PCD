import cv2
import torch
from facenet_pytorch import MTCNN
from PIL import Image

class FaceExtractor:
    def __init__(self, device='cpu', image_size=224, margin=20):
        """
        Initializes the MTCNN face detector.
        - keep_all=False: Only grabs the primary face in the frame.
        - margin: Adds padding around the face (crucial for boundary artifacts).
        """
        self.device = device
        self.mtcnn = MTCNN(
            image_size=image_size, 
            margin=margin, 
            keep_all=False, 
            post_process=False, 
            device=self.device
        )

    def extract_face_sequence(self, video_path, max_frames=32):
        """
        Reads a video, extracts frames, and returns a stacked tensor of cropped faces.
        Returns: Tensor of shape (Num_Frames, Channels, Height, Width)
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while cap.isOpened() and len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # OpenCV reads in BGR, but PyTorch/MTCNN expects RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            frames.append(pil_img)
            
        cap.release()

        if not frames:
            return None

        # MTCNN can process a batch (list) of PIL images efficiently
        cropped_faces = self.mtcnn(frames)

        # Filter out frames where MTCNN failed to find a face (returns None)
        valid_faces = [face for face in cropped_faces if face is not None]

        if not valid_faces:
            return None

        # Stack the individual face tensors into a single batch tensor
        # Normalizes pixel values automatically
        face_tensor = torch.stack(valid_faces)
        
        return face_tensor