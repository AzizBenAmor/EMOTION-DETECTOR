# download_model.py
import urllib.request
import os
import ssl

def download_pretrained_model():
    """Download a pre-trained emotion recognition model"""
    # This is a known pre-trained model for emotion recognition
    model_url = "https://github.com/oarriaga/face_classification/raw/master/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5"
    model_path = "emotion_model.h5"
    
    # Create a custom SSL context to avoid certificate issues
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    if not os.path.exists(model_path):
        print("Downloading pre-trained emotion model...")
        print("This may take a few minutes...")
        try:
            urllib.request.urlretrieve(model_url, model_path, context=ssl_context)
            print("Download completed successfully!")
            print(f"Model saved as: {model_path}")
        except Exception as e:
            print(f"Download failed: {e}")
            print("Please download the model manually from:")
            print("https://github.com/oarriaga/face_classification/raw/master/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5")
            print("And save it as 'emotion_model.h5' in the current directory")
    else:
        print("Model already exists!")
    
    return model_path

if __name__ == "__main__":
    download_pretrained_model()