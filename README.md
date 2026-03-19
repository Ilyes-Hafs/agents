# Local AI Project

## Overview

This project is a local AI assistant designed to handle various tasks such as text-to-speech (TTS), coding assistance, mathematical problem solving, web searches, file creation and conversion, and more. It leverages multiple open-source tools and models to provide comprehensive functionality.

## Full Stack

- **Ollama Models**: For language generation and processing.
  - `ollama/qwen2.5:3b`
  - `ollama/qwen2.5-coder:7b`
  - `ollama/deepseek-r1:7b`
  - `ollama/llama3.2:3b`

- **CrewAI**: A framework for managing and routing tasks to different agents.
- **Kokoro**: An ONNX model for text-to-speech synthesis.
- **SearXNG**: A privacy-respecting search engine.
- **ComfyUI**: A user interface for interacting with the AI.
- **Pandoc**: For file conversion between formats.

## External Dependencies

### Virtual Environments
- Located at `~/.venvs/agents`
  - Contains virtual environments for different components of the project.

### Ollama Models
- Installed in `~/.ollama`

### Docker Containers
- SearXNG, ComfyUI, and VSCodium + continue.dev are run using Docker containers.
  - Ensure Docker is installed and running on your system.

### Configuration Files
- `~/.aider.conf.yml`: Configuration file for the AIDER agent.

## Setup Instructions

### Prerequisites

1. **Install Python**: Ensure Python 3.8 or higher is installed.
2. **Install Docker**: Install Docker to run SearXNG, ComfyUI, and VSCodium + continue.dev containers.
3. **Clone the Repository**: Clone this repository to your local machine.

### Installation Steps

1. **Create Virtual Environments**:
   ```sh
   python -m venv ~/.venvs/agents
   source ~/.venvs/agents/bin/activate
   pip install -r requirements.txt
   ```

2. **Install Ollama Models**:
   - Download the models from the Ollama repository and place them in `~/.ollama`.

3. **Set Up Docker Containers**:
   ```sh
   docker run -d --name searxng -p 8888:8888 searxng/searxng
   docker run -d --name comfyui -p 5000:5000 -v ~/.aider.conf.yml:/app/.aider.conf.yml codercomfy/comfyui
   ```

4. **Install Pandoc**:
   ```sh
   sudo pacman -S pandoc
   ```

### Running the Project

1. **Activate Virtual Environment**:
   ```sh
   source ~/.venvs/agents/bin/activate
   ```

2. **Run the Main Script**:
   ```sh
   python main.py
   ```

3. **Interact with the AI**:
   - Type your questions or commands in the terminal.
   - The router will classify the request and route it to the appropriate agent.

### Configuration

- Edit `~/.aider.conf.yml` for AIDER agent configuration.
- Adjust Docker configurations as needed for SearXNG, ComfyUI, and VSCodium + continue.dev.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
