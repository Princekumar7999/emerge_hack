import numpy as np
import cv2
from PIL import ImageGrab
from ultralytics import YOLO
import time
import threading
from queue import Queue

class PersonBlur:
    def __init__(self):
        # Initialize YOLO model for person detection
        self.model = YOLO('yolov8n.pt')
        self.blur_strength = 45
        self.frame_queue = Queue(maxsize=2)  # Queue to store frames
        self.running = True

    def capture_screen(self):
        """Continuously capture screen and add to queue"""
        while self.running:
            screen = ImageGrab.grab()
            frame = np.array(screen)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if not self.frame_queue.full():
                self.frame_queue.put(frame)

            time.sleep(0.016)  # ~60 FPS

    def apply_body_blur(self, image, bbox):
        """Apply Gaussian blur to the body region"""
        x1, y1, x2, y2 = map(int, bbox)
        region = image[y1:y2, x1:x2]
        blurred = cv2.GaussianBlur(region, (self.blur_strength, self.blur_strength), 0)
        image[y1:y2, x1:x2] = blurred
        return image

    def process_frames(self):
        """Process frames from the queue"""
        try:
            # Start screen capture thread
            capture_thread = threading.Thread(target=self.capture_screen)
            capture_thread.start()

            while self.running:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get()

                    # Run YOLO detection
                    results = self.model(frame, stream=True)

                    # Process detections
                    for result in results:
                        boxes = result.boxes
                        for box in boxes:
                            # Check if detected object is a person (class 0)
                            if int(box.cls[0]) == 0:  # Ensure class is an integer
                                x1, y1, x2, y2 = box.xyxy[0]
                                frame = self.apply_body_blur(frame, (x1, y1, x2, y2))

                    # Display the frame
                    cv2.imshow('Real-time Person Detection', frame)

                    # Break loop on 'q' press
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.running = False
                        break
        finally:
            self.running = False
            capture_thread.join()
            cv2.destroyAllWindows()

def main():
    detector = PersonBlur()
    detector.process_frames()

if __name__ == "__main__":
    main()
