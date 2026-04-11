import requests
import argparse
import sys
import time
import json

# --- Config ---
BASE_URL = "http://localhost:8000"

def test_health():
    print("Checking API Health...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Body: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_predict(video_path):
    print(f"\nUploading video: {video_path}")
    url = f"{BASE_URL}/predict"
    
    try:
        with open(video_path, 'rb') as f:
            files = {'file': (video_path, f, 'video/mp4')}
            start_time = time.time()
            response = requests.post(url, files=files)
            end_time = time.time()
            
        print(f"HTTP Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("\n--- Results ---")
            print(f"Job ID: {data.get('job_id')}")
            print(f"Deepfake Probability: {data.get('deepfake_probability') * 100:.2f}%")
            print(f"Verdict: {data.get('verdict')}")
            print(f"Processing Time (server): {data.get('processing_time_ms')} ms")
            print(f"Total Time (total): {(end_time - start_time) * 1000:.0f} ms")
            
            print("\n--- Module Breakdown ---")
            branch_scores = data.get('branch_scores', {})
            for branch, score in branch_scores.items():
                print(f"{branch}: {score * 100:.2f}%")
        else:
            print(f"Error: {response.text}")
            
    except FileNotFoundError:
        print(f"Error: File '{video_path}' not found.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Deepfake Detection API")
    parser.add_argument("video_path", help="Path to the .mp4 video file")
    args = parser.parse_args()
    
    if test_health():
        test_predict(args.video_path)
    else:
        print("API Health Check Failed. Aborting.")
