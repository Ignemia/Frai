import services.chat.pipeline as chat_pipeline
import dotenv
import pathlib
import os

from services.database.connection import start_connection_pool
from services.cli.navigation import is_command, parse_command, handle_command

dotenv.load_dotenv(dotenv_path=pathlib.Path("local.env").resolve().absolute())

WELCOME_MESSAGE = "Hello! I'm your Personal Chatter companion, ready to chat and help. Simply type your message below to get started, or use /help to see available commands."

database_connection_pool = start_connection_pool()

def request_message(state):
    message = input(f"{state} > ")
    if is_command(message):
        command, args = parse_command(message) 
        handle_command(command, args)

def main():
    chat_pipeline.load_model()
    os.system("cls")
    print(WELCOME_MESSAGE)
    while True:
        request_message("home")

if __name__ == "__main__":
    main()