import cv2

class FrameExtractor:
    @staticmethod
    def extract_raw_frames(video_path, max_frames=32):
        """
        Extracts raw frames without face cropping.
        Returns a list of RGB numpy arrays.
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while cap.isOpened() and len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            
        cap.release()
        return frames