from services.cli.login_command import login_command
from services.cli.register_command import register_command

def is_command(message):
    if not message.startswith("/"):
        return False
    return True

def parse_command(message):
    if not is_command(message):
        return None, None
    
    parts = message[1:].split(" ", 1)
    command = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""
    
    return command, args


def help_command():
    print("Available commands: help, back, login, register, quit")

def quit_command():
    print("Exiting the application. Goodbye!")
    exit(0)

def back_command():
    print("Going back to the previous menu or state.")

def handle_command(command, args):
    match command:
        case "help" | "h" | "?" | "commands": 
            help_command()
        case "back" | "b":
            back_command()
        case "quit" | "exit" | "q":
            quit_command()
        case "login":
            login_command(args.strip())
        case "register":
            register_command(args.strip())
        case _:
            print(f"Unknown command: {command}")
            help_command()
