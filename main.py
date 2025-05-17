import services.chat.pipeline as chat_pipeline

def main():
    print("\nModel is loaded and ready to chat!")
    print("Enter your message (or 'quit' to exit):")
    
    while True:
        user_input = input("> ")
        if user_input.lower() in ["quit", "exit", "q"]:
            break
        
        print("\nAI friend: ", end="", flush=True)
        # Use streaming to see token-by-token generation
        chat_pipeline.send_query(
            message=user_input,
            stream=True,
            max_tokens=1024  # Increased token limit for more complete responses
        )
        print("\n")

if __name__ == "__main__":
    # Model is already loading when this module is imported
    main()