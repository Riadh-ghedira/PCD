import os
from moviepy.editor import VideoFileClip

class AudioExtractor:
    @staticmethod
    def extract_audio(video_path, output_dir="temp/"):
        """
        Extracts the audio track from a video and saves it as a .wav file.
        Returns the path to the saved audio file.
        """
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.basename(video_path).split('.')[0]
        output_audio_path = os.path.join(output_dir, f"{base_name}_audio.wav")

        try:
            video = VideoFileClip(video_path)
            
            # Check if video actually has an audio track
            if video.audio is None:
                video.close()
                return None
                
            # Write audio to file (logger=None mutes the console spam)
            video.audio.write_audiofile(output_audio_path, logger=None)
            video.close()
            
            return output_audio_path
            
        except Exception as e:
            print(f"Error extracting audio: {e}")
            return None