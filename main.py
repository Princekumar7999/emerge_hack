import numpy as np
import cv2
from PIL import ImageGrab
from ultralytics import YOLO
import time

class PersonBlur:
    def __init__(self):
        # Initialize YOLO model for person detection
        self.model = YOLO('yolov8n.pt')  # Using smallest YOLOv8 model for better performance
        
        # Gaussian blur parameters
        self.blur_strength = 45
        
    def apply_body_blur(self, image, bbox):
        """Apply Gaussian blur to the body region defined by bbox."""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Extract the region to blur
        region = image[y1:y2, x1:x2]
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(region, (self.blur_strength, self.blur_strength), 0)
        
        # Put the blurred region back
        image[y1:y2, x1:x2] = blurred
        
        return image
    
    def process_screen(self):
        """Main processing loop for screen capture and person detection."""
        try:
            while True:
                # Capture screen using PIL
                screen = ImageGrab.grab()
                frame = np.array(screen)
                
                # Convert from RGB (PIL) to BGR (OpenCV)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Run YOLO detection
                results = self.model(frame)
                
                # Process detections
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        # Check if detected object is a person (class 0 in COCO dataset)
                        if box.cls == 0:  
                            x1, y1, x2, y2 = box.xyxy[0]  # Get bounding box coordinates
                            # Apply blur to body region
                            frame = self.apply_body_blur(frame, (x1, y1, x2, y2))
                
                # Display the result
                cv2.imshow('Screen with Blurred Persons', frame)
                
                # Break loop on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # Small delay to prevent high CPU usage
                time.sleep(0.1)
                
        finally:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = PersonBlur()
    detector.process_screen()