# dataset_creator.py
import cv2
import os
import time
from datetime import datetime

class DatasetCreator:
    def __init__(self):
        self.emotions = ['happy', 'sad', 'angry', 'surprise', 'neutral']
        self.dataset_path = "my_emotion_dataset"
        self.create_folders()
    
    def create_folders(self):
        """Create folders for each emotion"""
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)
        
        for emotion in self.emotions:
            emotion_path = os.path.join(self.dataset_path, emotion)
            if not os.path.exists(emotion_path):
                os.makedirs(emotion_path)
        print("Created dataset folder structure")
    
    def capture_images(self, emotion, num_images=200):
        """Capture images for a specific emotion"""
        emotion_path = os.path.join(self.dataset_path, emotion)
        cap = cv2.VideoCapture(0)
        
        print(f"Capturing {num_images} images for emotion: {emotion}")
        print("Press 'c' to capture, 'q' to quit")
        
        count = 0
        while count < num_images:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Display instructions
            cv2.putText(frame, f"Emotion: {emotion}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Images captured: {count}/{num_images}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'c' to capture, 'q' to quit", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Dataset Creator', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                # Save image
                filename = f"{emotion}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{count}.jpg"
                filepath = os.path.join(emotion_path, filename)
                cv2.imwrite(filepath, frame)
                print(f"Saved: {filename}")
                count += 1
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"Completed capturing {count} images for {emotion}")

def main():
    creator = DatasetCreator()
    
    print("🎭 Emotion Dataset Creator")
    print("=" * 40)
    
    for i, emotion in enumerate(creator.emotions):
        print(f"{i+1}. {emotion}")
    
    print("\nWe'll capture images for each emotion.")
    input("Press Enter to start with the first emotion...")
    
    for emotion in creator.emotions:
        print(f"\n📸 Now capturing: {emotion.upper()}")
        print("Please make the corresponding facial expression")
        input("Press Enter when you're ready to start capturing...")
        
        creator.capture_images(emotion, num_images=50)  # Start with 50 per emotion
    
    print("\n✅ Dataset creation completed!")
    print(f"Dataset saved in: {creator.dataset_path}")

if __name__ == "__main__":
    main()