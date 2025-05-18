import services.chat.pipeline as chat_pipeline
import dotenv
import pathlib

dotenv.load_dotenv(dotenv_path=pathlib.Path("local.env").resolve().absolute())

def main():
    chat_pipeline.load_model()
    while True:
        user_input = input("> ")
        if user_input.lower() in ["quit", "exit", "q"]:
            break
        
        print("\nAI friend: ", end="", flush=True)
        chat_pipeline.send_query(
            message=user_input,
            stream=True,
            max_tokens=1024
        )
        print("\n")

if __name__ == "__main__":
    main()