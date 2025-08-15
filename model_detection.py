from ultralytics import YOLO
import cv2
import os

# Load the trained YOLOv8m model (replace 'path/to/your/trained_model.pt' with your actual model path)
model = YOLO('path/to/your/trained_model.pt')  # Assuming it's trained on sunflowers, class 0 is 'sunflower'

# Assume the drone camera stream is accessible via a video capture (e.g., simulator provides a video file or RTSP stream)
# For simulation, you might need to adjust the source. Here, using a video file as example; replace with your stream URL or camera index
video_source = 'path/to/drone_simulation_video.mp4'  # Or 0 for webcam, or RTSP URL for simulator stream
cap = cv2.VideoCapture(video_source)

# Set up tracking parameters
unique_sunflowers = set()  # To store unique track IDs of detected sunflowers
output_file = 'sunflower_count.txt'  # File to write the result

# Process the video stream frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run YOLOv8 tracking on the frame
    results = model.track(frame, persist=True, tracker="bytetrack.yaml")  # Use ByteTrack for tracking
    
    # Process detections
    for result in results:
        boxes = result.boxes
        if boxes.id is not None:  # If tracking IDs are available
            for i in range(len(boxes)):
                cls = int(boxes.cls[i])  # Class ID
                track_id = int(boxes.id[i])  # Track ID
                if cls == 0:  # Assuming class 0 is 'sunflower'; adjust if different
                    unique_sunflowers.add(track_id)  # Add unique track ID
    
    # Optional: Display the frame with annotations (for debugging)
    # annotated_frame = results[0].plot()
    # cv2.imshow('Drone Stream', annotated_frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Write the total unique count to file
with open(output_file, 'w') as f:
    f.write(f"Total unique sunflowers detected: {len(unique_sunflowers)}\n")

print(f"Processing complete. Results written to {output_file}")