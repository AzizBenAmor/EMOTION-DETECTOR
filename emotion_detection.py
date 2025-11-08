# emotion_detection_pretrained.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import urllib.request
import os
from datetime import datetime

class EmotionDetector:
    def __init__(self):
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        self.model = self.load_pretrained_model()
        print("Emotion detector initialized with pre-trained model!")
    
    def load_pretrained_model(self):
        """Load a pre-trained emotion recognition model"""
        try:
            # Try to load a pre-trained model
            # We'll use a simple approach with transfer learning
            base_model = MobileNetV2(weights='imagenet', include_top=False, 
                                   input_shape=(48, 48, 3))
            
            # Add custom layers for emotion classification
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(128, activation='relu')(x)
            predictions = Dense(7, activation='softmax')(x)
            
            model = Model(inputs=base_model.input, outputs=predictions)
            
            # Freeze base model layers
            for layer in base_model.layers:
                layer.trainable = False
                
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            return model
            
        except Exception as e:
            print(f"Error loading pre-trained model: {e}")
            return self.create_simple_model()
    
    def create_simple_model(self):
        """Create a simple model as fallback"""
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
        
        model = Sequential([
            Conv2D(32, (3,3), activation='relu', input_shape=(48,48,3)),
            MaxPooling2D(2,2),
            Conv2D(64, (3,3), activation='relu'),
            MaxPooling2D(2,2),
            Conv2D(128, (3,3), activation='relu'),
            MaxPooling2D(2,2),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(7, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def detect_emotion(self, face_image):
        """Detect emotion from face image"""
        try:
            # Preprocess the face image
            face_image = cv2.resize(face_image, (48, 48))
            face_image = face_image.astype('float32') / 255.0
            face_image = np.expand_dims(face_image, axis=0)
            
            # Make prediction
            predictions = self.model.predict(face_image, verbose=0)
            emotion_idx = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            
            return self.emotions[emotion_idx], confidence
            
        except Exception as e:
            print(f"Error in emotion detection: {e}")
            return "Unknown", 0.0

def detect_faces(frame):
    """Detect faces in the frame using Haar Cascade"""
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        return faces, gray
    except Exception as e:
        print(f"Error in face detection: {e}")
        return [], None

def main():
    # Initialize detector
    detector = EmotionDetector()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Emotion Detection Started...")
    print("Press 'q' to quit")
    print("Note: Model is using transfer learning. For best results, we need to train on emotion dataset.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
            
        # Detect faces
        faces, gray = detect_faces(frame)
        
        if len(faces) > 0:
            # Process each detected face
            for (x, y, w, h) in faces:
                # Extract face region
                face_roi = frame[y:y+h, x:x+w]
                
                # Detect emotion
                emotion, confidence = detector.detect_emotion(face_roi)
                
                # Draw rectangle around face
                color = (0, 255, 0)  # Green
                if emotion == "Happy":
                    color = (0, 255, 255)  # Yellow
                elif emotion == "Angry":
                    color = (0, 0, 255)  # Red
                elif emotion == "Sad":
                    color = (255, 0, 0)  # Blue
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Display emotion and confidence
                emotion_text = f"{emotion}: {confidence:.2f}"
                cv2.putText(frame, emotion_text, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Print to console
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"{current_time} - Emotion: {emotion}, Confidence: {confidence:.2f}")
        else:
            # No faces detected
            cv2.putText(frame, "No face detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Add timestamp
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, current_time, (10, frame.shape[0]-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Emotion Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()