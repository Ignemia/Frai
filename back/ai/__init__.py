from back.ai.chat import load_chat_model as _load_chat_model

def load_chat_model():
    """Load the chat model using the chat module."""
    return _load_chat_model()
    
def load_image_model():
    print("Image model loaded.")
    return True
    
def load_voice_activity_detection_model():
    print("Voice activity detection model loaded.")
    return True
    
def load_speech_recognition_model():
    print("Speech recognition model loaded.")
    return True
    
def load_text_to_speech_model():
    print("Text-to-speech model loaded.")
    return True