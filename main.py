import sys
import logging


from back.ai import load_chat_model, load_image_model, load_speech_recognition_model, load_text_to_speech_model, load_voice_activity_detection_model
from back.database import initiate_database_connection, test_database_connection
from front import initiate_cli, initiate_api
from orchestrator import initiate_orchestrator_layer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)
logger.info("Logging is set up.")

def main():
    try:
        # backend initiatior
        logger.info("Initiating database connection...")
        if not initiate_database_connection():
            logger.error("Failed to initiate database connection.")
            return 1
        logger.info("Database connection initiated.")
        logger.info("Testing database connection...")
        if not test_database_connection():
            logger.error("Database connection test failed.")
            return 1
        logger.info("Database connection test successful.")
        
        logger.info("Loading chat model...")
        if not load_chat_model():
            logger.error("Failed to load chat model.")
            return 1
        logger.info("Chat model loaded.")
        logger.info("Loading image model...")
        if not load_image_model():
            logger.error("Failed to load image model.")
            return 1
        logger.info("Image model loaded.")
        logger.info("Loading voice activity detection model...")
        if not load_voice_activity_detection_model():
            logger.error("Failed to load voice activity detection model.")
            return 1
        logger.info("Voice activity detection model loaded.")
        logger.info("Loading speech recognition model...")
        if not load_speech_recognition_model():
            logger.error("Failed to load speech recognition model.")
            return 1
        logger.info("Speech recognition model loaded.")
        logger.info("Loading text to speech model...")
        if not load_text_to_speech_model():
            logger.error("Failed to load text to speech model.")
            return 1
        logger.info("Text to speech model loaded.")
        
        # orchestrator initiator
        logger.info("Initiating orchestrator layer...")
        if not initiate_orchestrator_layer():
            logger.error("Failed to initiate orchestrator layer.")
            return 1
        logger.info("Orchestrator layer initiated.")
        
        # cli and api initiator
        logger.info("Initiating CLI...")
        if not initiate_cli():
            logger.error("Failed to initiate CLI.")
            return 1
        logger.info("CLI initiated.")
        logger.info("Initiating API...")
        if not initiate_api():
            logger.error("Failed to initiate API.")
            return 1
        logger.info("API initiated.")
        logger.info("Application started successfully.")
    except Exception as e:
        logger.error(f"An error occurred during application startup: {e}", exc_info=True)
        print(f"An error occurred: {e}")
        return 1
    return 0 

if __name__ == "__main__":
    return_code = main()
    sys.exit(return_code)