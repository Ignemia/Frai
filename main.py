import dotenv
dotenv.load_dotenv()

import logging
import os

from services.database.connection import start_connection_pool, validate_db_config, check_database_table_presence, initiate_tables
from services.cli.navigation import is_command, parse_command, handle_command

logging.basicConfig(level=logging.INFO)


WELCOME_MESSAGE = "Hello! I'm your Personal Chatter companion, ready to chat and help. Simply type your message below to get started, or use /help to see available commands."

database_connection_pool = None

def request_message(state):
    message = input(f"{state} > ")
    if is_command(message):
        command, args = parse_command(message) 
        handle_command(command, args)

def main():
    global database_connection_pool

    if not validate_db_config():
        print("Database configuration is invalid. Please check your .env file.")
        return 1
    database_connection_pool = start_connection_pool()
    if not check_database_table_presence():
        if not initiate_tables():
            print("Failed to create necessary database tables.")
            return 1
        print("Database tables are not present. Please check your database setup.")
    
    
    
    import services.chat.pipeline as chat_pipeline
    chat_pipeline.load_model()
    os.system("cls")
    print(WELCOME_MESSAGE)
    while True:
        request_message("home")

if __name__ == "__main__":
    main()