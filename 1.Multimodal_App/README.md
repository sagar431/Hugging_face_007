# Phi 3.5 Multimodal AI Assistant

## Overview

This project showcases a powerful multimodal AI assistant that combines the capabilities of Microsoft's Phi 3.5 models for text and vision, along with text-to-speech functionality. It demonstrates the potential of integrating multiple AI models to create a more comprehensive and interactive user experience.

![Phi 3.5 Multimodal AI Demo]

## Features

- **Text Interaction**: Utilizes the Phi-3.5-mini-instruct model for engaging and dynamic conversations.
- **Image Analysis**: Incorporates the Phi-3.5-vision-instruct model for advanced image processing and analysis.
- **Text-to-Speech**: Integrates Parler TTS for converting text responses to speech, enhancing the interactive experience.
- **User-Friendly Interface**: Built with Gradio for an intuitive and accessible user interface.
- **GPU Acceleration**: Supports CUDA for enhanced performance on compatible hardware.

## Tech Stack

- Python
- PyTorch
- Transformers (Hugging Face)
- Gradio
- Flash Attention 2
- Parler TTS
- NumPy
- Pillow (PIL)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/phi-3.5-multimodal-ai.git
   cd phi-3.5-multimodal-ai
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Install Flash Attention 2:
   ```
   pip install flash-attn --no-build-isolation
   ```

## Usage

1. Run the main script:
   ```
   python app.py
   ```

2. Open the provided URL in your web browser to access the Gradio interface.

3. Use the tabs to switch between text-based interaction and image analysis:
   - **Text Model**: Enter your message and adjust advanced options as needed.
   - **Vision Model**: Upload an image and ask questions about it.

## Configuration

- Adjust model parameters in the "Advanced Options" section of the text model interface.
- Modify system prompts and model configurations in the script for customized behavior.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- [Microsoft](https://www.microsoft.com/) for the Phi 3.5 models
- [Hugging Face](https://huggingface.co/) for the Transformers library
- [Gradio](https://www.gradio.app/) for the web interface framework
- [Anthropic](https://www.anthropic.com/) for inspiration and AI advancements

## Contact

For any queries or suggestions, please open an issue or contact [Your Name](mailto:your.email@example.com).

---

Happy coding! 🚀🤖