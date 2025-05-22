import logging
import base64
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding, hashes
from cryptography.hazmat.primitives.asymmetric import padding as rsa_padding
from cryptography.hazmat.backends import default_backend

from services.encryption.chat_crypto import (
    generate_rsa_key_pair, 
    serialize_key_pair,
    encrypt_chat_content, 
    decrypt_chat_content,
    encrypt_rsa_keys_with_credentials,
    decrypt_rsa_keys_with_credentials
)
from services.database.chat_utils import get_user_credentials

logger = logging.getLogger(__name__)

def prepare_encryption_data(chat_xml, user_id):
    """
    Prepare encryption data for a new chat.
    
    Returns:
        tuple: (key_pair, encrypted_data, encrypted_rsa_keys)
    """
    key_pair = generate_rsa_key_pair()
    serialized_keys = serialize_key_pair(key_pair)
    
    user_credentials = get_user_credentials(user_id)
    encrypted_rsa_keys = encrypt_rsa_keys_with_credentials(
        serialized_keys,
        user_credentials
    )
    
    encrypted_data = encrypt_chat_content(chat_xml, key_pair['public'])
    
    return key_pair, encrypted_data, encrypted_rsa_keys

def decrypt_user_rsa_keys(user_id, encrypted_rsa_keys):
    """Decrypt user's RSA keys using their credentials."""
    user_credentials = get_user_credentials(user_id)
    return decrypt_rsa_keys_with_credentials(
        encrypted_rsa_keys,
        user_credentials
    )

def decrypt_chat_with_keys(encrypted_content, encrypted_aes_key, aes_iv, rsa_private_key):
    """Decrypt chat content using keys."""
    return decrypt_chat_content(
        encrypted_content,
        encrypted_aes_key,
        aes_iv,
        rsa_private_key
    )

def encrypt_chat_content_with_existing_key(chat_xml, encrypted_aes_key, aes_iv, rsa_private_key):
    """
    Encrypt chat content using an existing AES key (which is itself encrypted with RSA).
    First decrypts the AES key with the RSA private key, then uses it to encrypt the chat content.
    """
    try:
        # Decrypt the AES key using the RSA private key
        aes_key = decrypt_aes_key_with_rsa(encrypted_aes_key, rsa_private_key)
        
        # Convert chat_xml to bytes if it's not already
        if isinstance(chat_xml, str):
            chat_xml = chat_xml.encode('utf-8')
            
        # Pad the data to match AES block size if needed
        padder = padding.PKCS7(algorithms.AES.block_size).padder()
        padded_data = padder.update(chat_xml) + padder.finalize()
        
        # Create an encryptor and encrypt the data
        cipher = Cipher(algorithms.AES(aes_key), modes.CBC(aes_iv), backend=default_backend())
        encryptor = cipher.encryptor()
        encrypted_content = encryptor.update(padded_data) + encryptor.finalize()
        
        # Encode the encrypted content to base64 for storage
        encoded_content = base64.b64encode(encrypted_content).decode('utf-8')
        
        return encoded_content
    except Exception as e:
        logger.error(f"Error encrypting chat content with existing key: {e}")
        raise

def decrypt_aes_key_with_rsa(encrypted_aes_key, rsa_private_key):
    """
    Decrypt an AES key that was encrypted with an RSA public key.
    """
    try:
        # Ensure encrypted_aes_key is in bytes format
        if isinstance(encrypted_aes_key, str):
            # Assuming it's base64 encoded if it's a string
            encrypted_aes_key = base64.b64decode(encrypted_aes_key)
        
        # Decrypt the AES key using the RSA private key
        decrypted_aes_key = rsa_private_key.decrypt(
            encrypted_aes_key,
            rsa_padding.OAEP(
                mgf=rsa_padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        logger.info("Successfully decrypted AES key with RSA private key")
        return decrypted_aes_key
    except Exception as e:
        logger.error(f"Error decrypting AES key: {e}", exc_info=True)
        raise
