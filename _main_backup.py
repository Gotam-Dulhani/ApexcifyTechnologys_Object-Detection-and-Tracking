import cv2
from ultralytics import YOLO

def main():
    # Load the YOLOv8 model
    print("Loading YOLOv8 model...")
    model = YOLO("yolov8n.pt")  # Use yolov8n.pt for faster, real-time tracking
    
    # Open the video capture
    # Use 0 for the default webcam, or provide a path to a video file
    video_source = 0
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source '{video_source}'")
        return
    
    print("Starting video capture. Press 'q' to quit.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame or video ended.")
            break
            
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)
        
        # Visualize the results on the frame
        # results[0].plot() draws boxes, labels, and track IDs on the image
        annotated_frame = results[0].plot()
        
        # Display the annotated frame
        cv2.imshow("YOLOv8 Object Tracking", annotated_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    print("Video capture ended.")

if __name__ == "__main__":
    main()
