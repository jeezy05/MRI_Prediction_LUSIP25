import os
import csv
import base64
from datetime import datetime, timedelta
from pathlib import Path
import requests
import json

def encode_image_to_base64(image_path):
    """Encode image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.standard_b64encode(image_file.read()).decode("utf-8")

def get_frame_description(image_path, ollama_url="http://localhost:11434"):
    """Get one-line description of frame using Moondream via Ollama."""
    try:
        image_data = encode_image_to_base64(image_path)
        
        payload = {
            "model": "moondream",
            "prompt": "Provide a single line description of what you see in this drone footage. Be concise and focus on objects, people, vehicles, or notable events.",
            "images": [image_data],
            "stream": False
        }
        
        response = requests.post(
            f"{ollama_url}/api/generate",
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        
        result = response.json()
        return result.get("response", "Unable to process frame").strip()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return "Unable to process frame"

def generate_telemetry_data(frame_id, num_frames):
    """Generate simulated telemetry data for a frame."""
    locations = ["Main Gate", "Garage", "North Perimeter", "South Perimeter", "Parking Lot"]
    altitudes = [50, 75, 100, 125, 150]
    
    location = locations[frame_id % len(locations)]
    altitude = altitudes[frame_id % len(altitudes)]
    
    base_time = datetime.strptime("00:00:00", "%H:%M:%S")
    frame_time = base_time + timedelta(seconds=frame_id * 2)
    time_str = frame_time.strftime("%H:%M:%S")
    
    return location, time_str, altitude

def process_frames_and_generate_csv(frames_dir, output_csv):
    """Process all frames and generate CSV with telemetry data."""
    frames_path = Path(frames_dir)
    frame_files = sorted(frames_path.glob("frame_*.jpg"))
    
    if not frame_files:
        print(f"No frames found in {frames_dir}")
        return
    
    print(f"Found {len(frame_files)} frames. Processing...")
    
    data = []
    for idx, frame_file in enumerate(frame_files):
        frame_id = frame_file.stem
        print(f"Processing {frame_id}... ({idx + 1}/{len(frame_files)})")
        
        description = get_frame_description(str(frame_file))
        location, time, altitude = generate_telemetry_data(idx, len(frame_files))
        
        data.append({
            "Frame ID": frame_id,
            "Frame Description": description,
            "Location": location,
            "Time": time,
            "Altitude (meters)": altitude
        })
    
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["Frame ID", "Frame Description", "Location", "Time", "Altitude (meters)"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    
    print(f"\nCSV file generated: {output_csv}")
    print(f"Total frames processed: {len(data)}")

if __name__ == "__main__":
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    frames_directory = project_root / "frames"
    output_file = project_root / "drone_telemetry.csv"
    
    process_frames_and_generate_csv(str(frames_directory), str(output_file))
