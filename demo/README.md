# Personal Chatter Demos

This directory contains demonstration scenarios that showcase the functionality of Personal Chatter.
These demos serve multiple purposes:
- Provide examples of how to use the system
- Act as functional tests to verify core capabilities
- Demonstrate the API and capabilities to new users

## Running Demos

You can run all demos using the demo runner:

```bash
python demo/run_demos.py
```

Or run a specific demo:

```bash
python demo/image_generation_demo.py
```

You can also use the test orchestrator:

```bash
python tests/test_orchestrator.py demo
```

## Available Demos

- `basic_chat_demo.py`: Demonstrates basic chat functionality
- `image_generation_demo.py`: Shows how to generate images with different parameters
- `memory_management_demo.py`: Demonstrates memory management capabilities
- `integration_demo.py`: Shows integration between chat and image generation
- `api_usage_demo.py`: Demonstrates how to use the API programmatically
