import time
import threading
from enum import Enum
from pathlib import Path
import sys
import cv2
import numpy as np
from ultralytics import YOLO
import importlib.util

import pyttsx3

import subprocess

# Add parent directory to path to import waverover control
sys.path.append(str(Path(__file__).parent.parent))

# Fix import for waverover-control.py
waverover_path = str(Path(__file__).parent.parent / 'waverover' / 'waverover-control.py')
spec = importlib.util.spec_from_file_location('waverover_control', waverover_path)
waverover_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(waverover_module)
WaveRoverControl = waverover_module.WaveRoverControl

class ObjectType(Enum):
    PAPER_BALL = "paper_ball"
    BOTTLE_CAP = "bottle_cap"

class SO101Workflow:
    def __init__(self, waverover_ip):
        """Initialize the workflow with WaveRover control."""
        self.waverover = WaveRoverControl(waverover_ip)
        self.current_object = None
        self.is_running = False
        self.workflow_thread = None

        # Initialize YOLOv8 classifier with ONNX model
        self.classifier = YOLO('../so101_classifier/weights/best.onnx')
        self.classifier.task = 'classify'  # Explicitly set task to suppress warning

        # TODO: Initialize SO101 arm controllers
        # self.so101_rover = SO101Controller()  # SO101 on WaveRover
        # self.so101_stationary = SO101Controller()  # Stationary SO101

    

    def speak(self, message):
        """Make the Mac say a message using the built-in TTS."""
        try:
            subprocess.run(['say', message])
        except Exception as e:
            print(f"Error using macOS voice: {e}")

    def speak_win(self, message):
        """Speak a message using Windows built-in TTS via pyttsx3."""
        try:
            engine = pyttsx3.init()
            engine.say(message)
            engine.runAndWait()
        except Exception as e:
            print(f"Speech error: {e}")

    def classify_object(self, image):
        """
        Classify the object in the image using the trained YOLOv8 model.
        Returns ObjectType.PAPER_BALL or ObjectType.BOTTLE_CAP
        """
        if image is None:
            raise ValueError("No image provided for classification")

        # Run inference on CPU
        results = self.classifier(image, device='cpu')

        # Get the top prediction
        top_result = results[0]
        class_id = int(top_result.probs.top1)
        confidence = float(top_result.probs.top1conf)

        print(f"Classification confidence: {confidence:.2f}")
        print("RESULTS:")
        print(results)

        # Map class ID to ObjectType
        if class_id == 1:  # paper_ball
            return ObjectType.PAPER_BALL
        else:  # bottle_cap
            return ObjectType.BOTTLE_CAP

    def capture_image(self):
        """
        Capture image from camera.
        Returns numpy array of the image.
        """
        # TODO: Implement actual camera capture
        return None  # Placeholder

    def move_waverover_to_position(self, position):
        """Move WaveRover to specified position."""
        if position == "A":
            print("Moving to position A")
            self.waverover.set_speed(0.3, 0.3)
            time.sleep(2)
            self.waverover.set_speed(0, 0)
        elif position == "B":
            print("Moving to position B")
            self.waverover.set_speed(0.3, 0.3)
            time.sleep(2)
            self.waverover.set_speed(0, 0)

    def so101_pickup(self, object_type):
        """Execute pickup sequence for SO101 arm on WaveRover."""
        print(f"SO101 on WaveRover picking up {object_type.value}")
        time.sleep(2)

    def so101_drop(self, object_type):
        """Execute drop sequence for SO101 arm on WaveRover."""
        print(f"SO101 on WaveRover dropping {object_type.value}")
        time.sleep(2)

    def so101_sort(self, object_type):
        """Execute sorting sequence for stationary SO101 arm."""
        if object_type == ObjectType.BOTTLE_CAP:
            print("Stationary SO101 sorting bottle cap to left bin")
        else:
            print("Stationary SO101 sorting paper ball to right bin")
        time.sleep(2)

    def workflow_loop(self):
        """Main workflow loop."""
        while self.is_running:
            try:
                print("\nStarting new workflow cycle...")

                self.move_waverover_to_position("A")

                image = self.capture_image()
                if image is not None:
                    self.current_object = self.classify_object(image)
                    print(f"Detected object: {self.current_object.value}")
                    self.so101_pickup(self.current_object)
                else:
                    print("Failed to capture image, skipping cycle")
                    continue

                self.move_waverover_to_position("B")
                self.so101_drop(self.current_object)
                self.so101_sort(self.current_object)

                print("Workflow cycle completed successfully")
                time.sleep(1)

            except Exception as e:
                print(f"Error in workflow: {e}")
                self.emergency_stop()
                break

    def start_workflow(self):
        """Start the workflow."""
        if not self.is_running:
            print("Starting workflow...")
            self.is_running = True
            self.workflow_thread = threading.Thread(target=self.workflow_loop)
            self.workflow_thread.daemon = True
            self.workflow_thread.start()

    def stop_workflow(self):
        """Stop the workflow gracefully."""
        print("Stopping workflow...")
        self.is_running = False
        if self.workflow_thread:
            self.workflow_thread.join()
        self.emergency_stop()

    def emergency_stop(self):
        """Emergency stop all systems."""
        print("Emergency stop activated")
        self.waverover.emergency_stop()
        # TODO: Add emergency stop for SO101 arms

    def capture_image_from_iphone(self):
        print("Waiting 1 second before capturing from iPhone camera...")
        time.sleep(1)
        cap = cv2.VideoCapture(0)  # 0 should be your iPhone if it's selected as the default camera
        if not cap.isOpened():
            raise RuntimeError("Could not open iPhone camera")
        ret, frame = cap.read()
        if ret:
            filename = "captured_iphone.jpg"
            cv2.imwrite(filename, frame)
            print(f"Image saved to {filename}")
            cap.release()
            cv2.destroyAllWindows()
            return frame
        else:
            print("Failed to capture image")
            cap.release()
            cv2.destroyAllWindows()
            return None

    def test_classification_and_drive(self, image_path=None, use_iphone_camera=False):
        """
        Test method: Classifies an image and drives the WaveRover based on the result.
        If paper ball: drive forward. If bottle cap: drive backward.
        If use_iphone_camera is True, capture from iPhone camera.
        """
        if use_iphone_camera:
            image = self.capture_image_from_iphone()
            if image is None:
                print("Failed to capture image from iPhone camera.")
                return
        elif image_path is not None:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image from {image_path}")
                return
        else:
            image = self.capture_image()
            if image is None:
                print("Failed to capture image from camera.")
                return

        # Save the image being classified
        cv2.imwrite("test_image.jpg", image)
        print("Saved the image being classified as 'test_image.jpg'")

        detected_object = self.classify_object(image)
        print(f"Test detected object: {detected_object.value}")

        if detected_object == ObjectType.PAPER_BALL:
            print("Driving forward (paper ball detected)...")
            self.speak_win("paper ball detected")
            time.sleep(7)
            self.waverover.set_speed(-0.2, -0.2)
            time.sleep(3)
            self.waverover.set_speed(0, 0)
        elif detected_object == ObjectType.BOTTLE_CAP:
            print("Driving backward (bottle cap detected)...")
            self.speak_win(" bottle cap detected")
            time.sleep(5)
            self.waverover.set_speed(-0.2, -0.2)
            time.sleep(2)
            self.waverover.set_speed(0, 0)
        else:
            print("Unknown object detected. No movement.")
        
        time.sleep(4)
        self.waverover.set_speed(0.2, 0.2)
        time.sleep(2.5)
        self.waverover.set_speed(0, 0)


def main():
    workflow = SO101Workflow("192.168.8.127")  # Replace with actual IP

    try:
        print("Starting workflow...")
        # Uncomment below to run full workflow loop
        # workflow.start_workflow()

        # Test mode: single classification + movement
        # To use iPhone camera, set use_iphone_camera=True
        #workflow.test_classification_and_drive(use_iphone_camera=True)

        test_image_path = "C:/Users/Zakariea/Documents/GitHub/lerobot/paper_ball_1.jpeg"
        workflow.test_classification_and_drive(test_image_path)

    
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received, stopping workflow...")
        workflow.stop_workflow()
    finally:
        workflow.emergency_stop()

if __name__ == "__main__":
    main()
