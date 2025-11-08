# real_emotion_detection.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
from datetime import datetime

class RealEmotionDetector:
    def __init__(self):
        # Emotion labels for FER2013 dataset
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.model = self.load_real_model()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        if self.model:
            print("Real emotion detector initialized successfully!")
        else:
            print("Emotion detector initialized with limited functionality.")
    
    def load_real_model(self):
        """Load the actual pre-trained emotion model"""
        model_path = "emotion_model.h5"
        
        if not os.path.exists(model_path):
            print("❌ Pre-trained model not found!")
            return None
            
        try:
            print("Loading pre-trained emotion model...")
            model = load_model(model_path, compile=False)
            print("✅ Pre-trained model loaded successfully!")
            return model
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None
    
    def detect_emotion(self, face_image):
        """Detect emotion using pre-trained model"""
        if self.model is None:
            return "Model not loaded", 0.0
            
        try:
            # Preprocess the face image for the specific model
            # The model expects grayscale images of size 64x64
            gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            gray_face = cv2.resize(gray_face, (64, 64))
            gray_face = gray_face.astype('float32') / 255.0
            gray_face = np.expand_dims(gray_face, axis=0)
            gray_face = np.expand_dims(gray_face, axis=-1)  # Add channel dimension
            
            # Make prediction
            predictions = self.model.predict(gray_face, verbose=0)
            emotion_idx = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            
            return self.emotions[emotion_idx], confidence
            
        except Exception as e:
            print(f"Error in emotion detection: {e}")
            return "Error", 0.0

def main():
    # Initialize detector
    detector = RealEmotionDetector()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("\n🎭 Real Emotion Detection Started!")
    print("=====================================")
    print("Press 'q' to quit")
    print("Press 's' to save current frame")
    print("=====================================")
    
    # Emotion statistics
    emotion_count = {emotion: 0 for emotion in detector.emotions}
    emotion_history = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
            
        # Create a copy for display
        display_frame = frame.copy()
        
        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            # Process each detected face
            for (x, y, w, h) in faces:
                # Extract face region with padding
                padding = 20
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(frame.shape[1], x + w + padding)
                y2 = min(frame.shape[0], y + h + padding)
                face_roi = frame[y1:y2, x1:x2]
                
                if face_roi.size == 0:
                    continue
                
                # Detect emotion
                emotion, confidence = detector.detect_emotion(face_roi)
                
                if emotion != "Error" and emotion != "Model not loaded":
                    emotion_count[emotion] += 1
                    emotion_history.append((emotion, confidence, datetime.now()))
                
                # Choose color based on emotion
                color_map = {
                    'Happy': (0, 255, 255),    # Yellow
                    'Angry': (0, 0, 255),      # Red
                    'Sad': (255, 0, 0),        # Blue
                    'Surprise': (0, 255, 0),   # Green
                    'Neutral': (255, 255, 255), # White
                    'Fear': (128, 0, 128),     # Purple
                    'Disgust': (0, 128, 128),   # Teal
                    'Error': (128, 128, 128),   # Gray
                    'Model not loaded': (128, 128, 128) # Gray
                }
                color = color_map.get(emotion, (0, 255, 0))
                
                # Draw rectangle around face
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 3)
                
                # Display emotion and confidence with background for better visibility
                text = f"{emotion}: {confidence:.2f}"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(display_frame, (x, y-text_size[1]-10), (x+text_size[0], y), color, -1)
                cv2.putText(display_frame, text, (x, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)  # Black text
                
                # Print to console (less frequent to avoid spam)
                current_time = datetime.now()
                if len(emotion_history) == 0 or (current_time - emotion_history[-1][2]).seconds >= 2:
                    print(f"{current_time.strftime('%H:%M:%S')} - Emotion: {emotion}, Confidence: {confidence:.2f}")
        else:
            # No faces detected
            cv2.putText(display_frame, "No face detected - Move closer to camera", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display emotion statistics
        stats_y = 60
        active_emotions = [(emotion, count) for emotion, count in emotion_count.items() if count > 0]
        active_emotions.sort(key=lambda x: x[1], reverse=True)
        
        cv2.putText(display_frame, "Emotion Stats:", (10, stats_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        stats_y += 25
        
        for emotion, count in active_emotions[:4]:  # Show top 4
            color = color_map.get(emotion, (255, 255, 255))
            cv2.putText(display_frame, f"{emotion}: {count}", (10, stats_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            stats_y += 20
        
        # Add timestamp
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(display_frame, current_time, (10, display_frame.shape[0]-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add help text
        cv2.putText(display_frame, "Press 'q' to quit | 's' to save", 
                   (display_frame.shape[1]-300, display_frame.shape[0]-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.imshow('Real Emotion Detection', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save current frame
            filename = f"emotion_capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, frame)
            print(f"📸 Frame saved as: {filename}")
            
    cap.release()
    cv2.destroyAllWindows()
    
    # Print final statistics
    print("\n" + "="*50)
    print("FINAL EMOTION STATISTICS:")
    print("="*50)
    total_detections = sum(emotion_count.values())
    if total_detections > 0:
        for emotion, count in emotion_count.items():
            if count > 0:
                percentage = (count / total_detections) * 100
                print(f"{emotion}: {count} times ({percentage:.1f}%)")
    else:
        print("No emotions detected during the session.")
    
    print("Session ended.")

if __name__ == "__main__":
    main()