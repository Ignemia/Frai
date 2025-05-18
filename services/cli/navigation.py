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
    print("Available commands: help, back, quit")

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
        case _:
            print(f"Unknown command: {command}")
            help_command()
