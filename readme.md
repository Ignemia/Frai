# Personal Chatter

## Purpose
Personal Chatter is a user-centric text bot designed to remember information about you and learn about your preferences to become a supportive companion.

## Why?
In modern day we have access to many AI models and chatbots but if you want it to remember anything about yourself you will have to pay or share the information with the company.  
Neither of which sound good for people with financial problems or security oriented people.  
Therefore I want to provide an open-source alternative that will allow users to have their own instance of their chatbot and run it at home, where their data will be stored on their own device. Purely at the expense of their own electricity.

## Implementation

### Models Used
-   **Voice Processing**: `pyannote/segmentation-3.0` + `pyannote/speaker-diarization-3.1`
-   **Vector DB Embeddings**: `google/Gemma-Embeddings-v1.0`
-   **Text Generation**: `google/gemma-3-4b-it` (this model can be replaced with other compatible models)
-   **Image Generation**: `black-forest-labs/FLUX.1-dev`
-   **Sentiment analysis**: `tabularisai/multilingual-sentiment-analysis`

## How It Works
The basic workflow is as follows:
1.  **Login & Initial Setup**: The user logs into the application.
2.  **User Profiling**: The user is prompted to share some basic personal information.
3.  **Bot Interaction**:
    *   The bot aims to be helpful and informative. For example, it might share the etymological meaning of your name or insights based on your date of birth according to astrology.
    *   The bot also shares some information about itself to build rapport.
4.  **Chat Creation**: Users can create as many chat sessions as they wish.
5.  **Information Storage**: When new information about the user is shared, it is extracted and stored locally on the device.
6.  **Information Retrieval & Search**: If the bot needs to explain something or find information, it self-prompts and performs an online search using the Brave Search API.
7.  **Image Generation**: If a user requests an image, their prompt is parsed and refined based on preconfigured preferences, then sent to the `flux.1` image generation pipeline.

All these processes run asynchronously on the same host device.

## Minimum System Requirements
To run Personal Chatter effectively with all its features, your system should meet the following minimum specifications:
-   **CPU**: Modern multi-core (6+ cores) processor (e.g., Intel Core i5 13th gen / AMD Ryzen 5000 series or newer).
-   **RAM**: 32GB (64GB recommended for smoother performance).
-   **GPU**: NVIDIA GPU with at least 12GB of VRAM (e.g., RTX 4070 or better). This is important for image generation and can accelerate text model performance.
-   **Storage**: SSD with at least 100GB of free space for models, application data, and user profiles.
-   **Operating System**: Windows, Linux, or macOS (ensure compatibility of all model dependencies).

## Installation
1. **Clone the repository**  
    HTTPS
    ```bash
    git clone https://github.com/Ignemia/personal-chatter.git --recursive
    cd personal-chatter
    ```  

    SSH
    ```bash
    git clone git@github.com:Ignemia/personal-chatter.git --recursive
    cd personal-chatter
    ```

2.  **Install dependencies**:
    It's recommended to use a virtual environment.  
    1. Install CUDA or ROCm
        - [cuda](https://developer.nvidia.com/cuda-toolkit)
        - [ROCm](https://www.amd.com/en/products/software/rocm.html)
    2. Install torch based on your device specs: [guide](https://pytorch.org/get-started/locally/)  
    3. 
        ```bash
        python -m venv .venv
        # On Windows:
        # .venv\Scripts\activate
        # On macOS/Linux:
        # source .venv/bin/activate
        pip install -r requirements.txt
        ```

## Running the Application
1.  **Navigate to the project directory** (if you are not already there).
2.  **Activate your virtual environment** (if you created one).
3.  **Run the main script**:
    ```bash
    python main.py
    ```

---
*Note: This is a basic outline. Specific model setup, API key configurations (e.g., for Brave Search API), and detailed dependencies might require further documentation within the project.*
