import logging

from orchestrator.chat import initiate_chat_orchestrator
from orchestrator.chatmod import initiate_chat_moderator
from orchestrator.img2img import initiate_image_to_image_orchestrator
from orchestrator.text2img import initiate_text_to_image_orchestrator
from orchestrator.users import initiate_user_actions_orchestrator
from orchestrator.voicein import initiate_speech_recognition_orchestrator, initiate_voice_activity_detection_orchestrator
from orchestrator.voiceout import initiate_text_to_speech_orchestrator

logger = logging.getLogger(__name__)

def initiate_orchestrator_layer():
    logger.info("Initiating orchestrator layer...")

    orchestrators = [
        ("chat_orchestrator", initiate_chat_orchestrator),
        ("chat_moderator", initiate_chat_moderator),
        ("image_to_image_orchestrator", initiate_image_to_image_orchestrator),
        ("text_to_image_orchestrator", initiate_text_to_image_orchestrator),
        ("user_actions_orchestrator", initiate_user_actions_orchestrator),
        ("voice_activity_detection_orchestrator", initiate_voice_activity_detection_orchestrator),
        ("speech_recognition_orchestrator", initiate_speech_recognition_orchestrator),
        ("text_to_speech_orchestrator", initiate_text_to_speech_orchestrator),
    ]

    for name, func in orchestrators:
        logger.info(f"Initiating {name}...")
        try:
            result = func()
            if result is False or result is None:
                logger.error(f"Failed to initiate {name}.")
                return False
            logger.info(f"{name} initiated successfully.")
        except Exception as e:
            logger.error(f"Error initiating {name}: {e}", exc_info=True)
            return False

    logger.info("Orchestrator layer initiated successfully.")
    return True
