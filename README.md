Person-Blur-System

A real-time system designed to protect privacy by blurring individuals detected in live screen captures, leveraging cutting-edge AI techniques.

🚀 Inspiration

The increasing use of screen recording in public and private settings often leads to privacy concerns. This project addresses the need to anonymize individuals in real-time during screen captures, ensuring privacy protection in a fast and efficient manner.

🧠 What it does

The Person-Blur-System captures your screen in real-time and automatically blurs detected individuals. Key features include:

Real-time person detection: Utilizes YOLOv8 for detecting people in live screen captures.

Gaussian blur application: Ensures privacy by applying adjustable Gaussian blur to detected individuals.

Screen capture integration: Processes live screen captures directly.

Customizable settings: Allows adjustments for blur strength and detection sensitivity.

⚙️ How we built it

The system was built using the following technologies and tools:

Frontend: OpenCV for displaying real-time processed frames.

Backend: Python with YOLOv8 for object detection and PIL for screen capturing.

AI Models: YOLOv8n for fast and efficient detection.

High-level Workflow:

Screen is captured using PIL's ImageGrab.

YOLOv8 detects human figures in the captured frames.

Gaussian blur is applied to regions where persons are detected.

The processed frame is displayed in real-time.

💡 Challenges we ran into

Real-time performance: Ensuring low-latency processing for smooth frame display.

Model accuracy: Tuning YOLOv8 for detecting individuals accurately in various lighting and screen conditions.

Resource optimization: Managing high CPU and memory usage during live processing.

We optimized performance by using the smallest YOLOv8 model and implementing a delay to reduce resource consumption.

🏆 Accomplishments that we're proud of

Achieved seamless real-time detection and blurring on live screen captures.

Successfully integrated YOLOv8 with OpenCV for efficient frame processing.

Delivered a functional system that prioritizes privacy and usability.

🧪 What we learned

How to implement real-time screen capture and processing using PIL and OpenCV.

Efficient integration of object detection models like YOLOv8 into live applications.

Techniques for optimizing Gaussian blur for performance and quality.

🔮 What's next for Person-Blur-System

Enhancing detection accuracy: Fine-tuning the model for improved detection in complex scenarios.

Customizable regions: Allow users to select specific areas of the screen to process.

Cloud deployment: Enabling remote processing for resource-limited devices.

Mobile integration: Developing a mobile app version for on-device screen privacy.

💻 Tech Stack

Screen Capture: PIL (ImageGrab)

Detection Model: YOLOv8n

Backend: Python, OpenCV

Frameworks: TensorFlow (for YOLOv8 integration)

🛠️ Installation and Usage

Clone the repository:

git clone https://github.com/Princekumar7999/emerge_hack.git

Install dependencies:

Ensure you have Python 3.8+ installed, then run:

pip install -r requirements.txt

Run the application:

Navigate to the project directory and execute the script:

python person_blur.py

Usage:

Run the script to start the real-time screen capture and processing.

View the processed screen with blurred individuals in the OpenCV display window.

Press q to exit the application.
