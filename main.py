from api.api import start_backend_api


def main():
    try:
        start_backend_api()

        load_image_pipeline()
        load_embeddings_pipeline()
        load_text_pipeline()
        load_audio_pipeline()

    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    main()