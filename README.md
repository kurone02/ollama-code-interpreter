# Code interpreter

## Dependencies
### Install ollama
- Refer to [this link](https://ollama.com/download) to install ollama
- Or (more preferably) install with docker with [the following tutorial](https://ollama.com/blog/ollama-is-now-available-as-an-official-docker-image)
    - CPU only:
    ```bash
    docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
    ```
    - With NVIDIA GPU (Remember to install the Nvidia container toolkit):
    ```bash
    docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
    ```
    - If you want to run ollama with specific NVIDIA GPU:
    ```bash
    docker run -d --gpus='"device=[gpu devices, e.g., 0,1,3 or all]"' -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
    ```
    - Run the model with
    ```bash
    docker exec -it ollama ollama run [model_name, e.g., deepseek-coder:33b-instruct-q8_0]
    ```