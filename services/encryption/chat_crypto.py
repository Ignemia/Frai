import os
import base64
import logging
import hashlib
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

logger = logging.getLogger(__name__)

def generate_rsa_key_pair():
    """Generate a new RSA key pair for chat encryption"""
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )
    public_key = private_key.public_key()
    
    return {
        'private': private_key,
        'public': public_key
    }

def serialize_key_pair(key_pair):
    """Serialize RSA key pair for storage"""
    private_pem = key_pair['private'].private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    
    public_pem = key_pair['public'].public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    
    return {
        'private': private_pem.decode('utf-8'),
        'public': public_pem.decode('utf-8')
    }

def deserialize_key_pair(serialized_keys):
    """
    Deserialize RSA key pair from storage.
    Can deserialize from just private key PEM if public is not provided.
    """
    private_key = serialization.load_pem_private_key(
        serialized_keys['private'].encode('utf-8'),
        password=None,
        backend=default_backend()
    )
    
    public_key_pem = serialized_keys.get('public')
    if public_key_pem:
        public_key = serialization.load_pem_public_key(
            public_key_pem.encode('utf-8'),
            backend=default_backend()
        )
    else:
        # Derive public key from private key if not provided
        public_key = private_key.public_key()
    
    return {
        'private': private_key,
        'public': public_key
    }

def encrypt_chat_content(content, public_key):
    """Encrypt chat content with RSA public key"""
    # For large content, use hybrid encryption: AES for content, RSA for AES key
    aes_key = os.urandom(32)  # 256-bit key
    iv = os.urandom(16)  # 128-bit IV
    
    # Encrypt content with AES
    cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    
    # Pad content to AES block size (16 bytes)
    padded_content = content.encode('utf-8')
    padding_length = 16 - (len(padded_content) % 16)
    padded_content += bytes([padding_length]) * padding_length
    
    encrypted_content = encryptor.update(padded_content) + encryptor.finalize()
    
    # Encrypt AES key with RSA
    encrypted_key = public_key.encrypt(
        aes_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    
    return {
        'content': base64.b64encode(encrypted_content).decode('utf-8'),
        'encrypted_key': base64.b64encode(encrypted_key).decode('utf-8'),
        'iv': base64.b64encode(iv).decode('utf-8')
    }

def decrypt_chat_content(encrypted_data, private_key):
    """
    Decrypt chat content with RSA private key.
    encrypted_data is a dict: {'content': b64_aes_encrypted_xml, 
                               'encrypted_key': b64_rsa_encrypted_aes_key, 
                               'iv': b64_aes_iv}
    private_key is an RSA private key object.
    """
    # Decode base64 values
    encrypted_content = base64.b64decode(encrypted_data['content'])
    encrypted_key = base64.b64decode(encrypted_data['encrypted_key'])
    iv = base64.b64decode(encrypted_data['iv'])
    
    # Decrypt AES key with RSA
    aes_key = private_key.decrypt(
        encrypted_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    
    # Decrypt content with AES
    cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    
    padded_content = decryptor.update(encrypted_content) + decryptor.finalize()
    
    # Remove padding
    padding_length = padded_content[-1]
    content = padded_content[:-padding_length]
    
    return content.decode('utf-8')

def encrypt_rsa_keys_with_credentials(serialized_keys, username, password_hash):
    """
    Encrypt RSA private key PEM string using a derived key from username and password hash.
    'serialized_keys' dict is expected to have 'private' key with PEM string.
    """
    # Create a key from the username and password hash
    key_material = f"{username}:{password_hash}".encode('utf-8')
    derived_key = hashlib.sha256(key_material).digest()
    iv = os.urandom(16)
    
    # Serialize and encrypt the RSA private key string
    private_key_pem_str = serialized_keys['private'] # Expecting 'private' to be the PEM string
    
    # Encrypt with AES
    cipher = Cipher(algorithms.AES(derived_key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    
    # Pad content
    padded_content = private_key_pem_str.encode('utf-8') # Changed keys_str to private_key_pem_str
    padding_length = 16 - (len(padded_content) % 16)
    padded_content += bytes([padding_length]) * padding_length
    
    encrypted_keys = encryptor.update(padded_content) + encryptor.finalize()
    
    return {
        'encrypted_key': base64.b64encode(encrypted_keys).decode('utf-8'),
        'iv': base64.b64encode(iv).decode('utf-8')
    }

def decrypt_rsa_keys_with_credentials(encrypted_data, username, password_hash):
    """
    Decrypt RSA private key PEM string using credentials.
    'encrypted_data' is a dict: {'encrypted_key': b64_encrypted_private_key_pem, 'iv': b64_iv}
    Returns a dict: {'private': private_key_pem_string}
    """
    # Recreate the derived key
    key_material = f"{username}:{password_hash}".encode('utf-8')
    derived_key = hashlib.sha256(key_material).digest()
    
    # Decode base64 values
    encrypted_keys = base64.b64decode(encrypted_data['encrypted_key'])
    iv = base64.b64decode(encrypted_data['iv'])
    
    # Decrypt with AES
    cipher = Cipher(algorithms.AES(derived_key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    
    padded_content = decryptor.update(encrypted_keys) + decryptor.finalize()
    
    # Remove padding
    padding_length = padded_content[-1]
    content = padded_content[:-padding_length]
    
    # Deserialize the RSA private key string
    private_key_str = base64.b64decode(content).decode('utf-8')
    
    return {
        'private': private_key_str,
        # Public key is not decrypted here, it can be derived from the private key object later
    }
