import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Uuid
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.dialects.postgresql import UUID as PG_UUID # Use a distinct alias
import uuid as py_uuid # Standard Python UUID library

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    user_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(Text, nullable=False, unique=True)

    password_entry = relationship("PasswordEntry", back_populates="user", uselist=False, cascade="all, delete-orphan")
    sessions = relationship("Session", back_populates="user", cascade="all, delete-orphan")
    chats = relationship("Chat", back_populates="user", cascade="all, delete-orphan")
    user_keys = relationship("UserKey", back_populates="user", uselist=False, cascade="all, delete-orphan")
    user_keys = relationship("UserKey", back_populates="user", uselist=False, cascade="all, delete-orphan")

class PasswordEntry(Base):
    __tablename__ = "passwords"
    user_id = Column(Integer, ForeignKey("users.user_id", ondelete="CASCADE"), primary_key=True)
    hashed_password = Column(String(64), nullable=False)
    expire_date = Column(DateTime, nullable=False)

    user = relationship("User", back_populates="password_entry")

class Chat(Base):
    __tablename__ = "chats"
    chat_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    chat_name = Column(Text, nullable=False, default="Untitled Chat")
    contents = Column(Text, nullable=False, default='') # Stores application-encrypted chat content
    last_modified = Column(DateTime, default=datetime.datetime.now, onupdate=datetime.datetime.now)

    user = relationship("User", back_populates="chats")
    key = relationship("ChatKey", back_populates="chat", uselist=False, cascade="all, delete-orphan")

class ChatKey(Base):
    __tablename__ = "chat_keys"
    chat_id = Column(Integer, ForeignKey("chats.chat_id", ondelete="CASCADE"), primary_key=True)
    encrypted_key = Column(Text, nullable=False) # ChatAESKey, encrypted with the chat's RSA public key
    iv = Column(Text, nullable=False) # IV for ChatAESKey encryption of content

    # New columns for storing the chat's RSA private key, encrypted by UserDerivedKey
    encrypted_rsa_private_key = Column(Text, nullable=False)
    user_derived_key_iv = Column(Text, nullable=False) # IV for UserDerivedKey encryption of RSA private key

    chat = relationship("Chat", back_populates="key")

class Session(Base):
    __tablename__ = "sessions"
    # Use PG_UUID for PostgreSQL specific UUID type, and python's uuid for default generation
    session_id = Column(PG_UUID(as_uuid=True), primary_key=True, default=py_uuid.uuid4)
    user_id = Column(Integer, ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.now, nullable=False)
    expires_at = Column(DateTime, nullable=False)

    user = relationship("User", back_populates="sessions")

class UserKey(Base):
    __tablename__ = "user_keys"
    user_id = Column(Integer, ForeignKey("users.user_id", ondelete="CASCADE"), primary_key=True)
    encrypted_keys = Column(Text, nullable=False)  # Encrypted RSA private key data
    
    user = relationship("User", back_populates="user_keys")
