import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, TextIteratorStreamer, BitsAndBytesConfig
import gradio as gr
from threading import Thread
import numpy as np
from PIL import Image
import subprocess
import spaces
from parler_tts import ParlerTTSForConditionalGeneration
import soundfile as sf
import tempfile
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Add this global variable after the imports
executor = ThreadPoolExecutor(max_workers=2)

# Install flash-attention
subprocess.run('pip install flash-attn --no-build-isolation', env={'FLASH_ATTENTION_SKIP_CUDA_BUILD': "TRUE"}, shell=True)

# Constants
TITLE = "<h1><center>Phi 3.5 Multimodal (Text + Vision)</center></h1>"
DESCRIPTION = "# Phi-3.5 Multimodal Demo (Text + Vision)"

# Model configurations
TEXT_MODEL_ID = "microsoft/Phi-3.5-mini-instruct"
VISION_MODEL_ID = "microsoft/Phi-3.5-vision-instruct"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Quantization config for text model
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Load models and tokenizers
text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_ID)
text_model = AutoModelForCausalLM.from_pretrained(
    TEXT_MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=quantization_config
)

try:
    vision_model = AutoModelForCausalLM.from_pretrained(
        VISION_MODEL_ID, 
        trust_remote_code=True, 
        torch_dtype="auto", 
        attn_implementation="flash_attention_2"
    ).to(device).eval()
except Exception as e:
    print(f"Error loading model with flash attention: {e}")
    print("Falling back to default attention implementation")
    vision_model = AutoModelForCausalLM.from_pretrained(
        VISION_MODEL_ID, 
        trust_remote_code=True, 
        torch_dtype="auto"
    ).to(device).eval()

vision_processor = AutoProcessor.from_pretrained(VISION_MODEL_ID, trust_remote_code=True)

# Initialize Parler-TTS
tts_model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(device)
tts_tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")

# Add the generate_speech function here
async def generate_speech(text, tts_model, tts_tokenizer):
    tts_input_ids = tts_tokenizer(text, return_tensors="pt").input_ids.to(device)
    tts_description = "A clear and natural voice reads the text with moderate speed and expression."
    tts_description_ids = tts_tokenizer(tts_description, return_tensors="pt").input_ids.to(device)
    
    with torch.no_grad():
        audio_generation = tts_model.generate(input_ids=tts_description_ids, prompt_input_ids=tts_input_ids)
    
    return audio_generation.cpu().numpy().squeeze()

from gradio import Error as GradioError

@spaces.GPU(timeout=300)
def stream_text_chat(message, history, system_prompt, temperature=0.8, max_new_tokens=1024, top_p=1.0, top_k=20, use_tts=True):
    try:
        conversation = [{"role": "system", "content": system_prompt}]
        for prompt, answer in history:
            conversation.extend([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ])
        conversation.append({"role": "user", "content": message})

        input_ids = text_tokenizer.apply_chat_template(conversation, add_generation_prompt=True, return_tensors="pt").to(text_model.device)
        attention_mask = torch.ones_like(input_ids)
        streamer = TextIteratorStreamer(text_tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)

        generate_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            eos_token_id=text_tokenizer.eos_token_id,
            pad_token_id=text_tokenizer.pad_token_id,
            streamer=streamer,
        )

        thread = Thread(target=text_model.generate, kwargs=generate_kwargs)
        thread.start()

        buffer = ""
        for new_text in streamer:
            buffer += new_text
            yield history + [[message, buffer]], None  # Yield None for audio initially

        # Only attempt TTS if it's enabled and we have a response
        if use_tts and buffer:
            try:
                audio = generate_speech_sync(buffer, tts_model, tts_tokenizer)
                yield history + [[message, buffer]], (tts_model.config.sampling_rate, audio)
            except Exception as e:
                print(f"TTS failed: {str(e)}")
                yield history + [[message, buffer]], None
        else:
            yield history + [[message, buffer]], None

    except GradioError:
        yield history + [[message, "GPU task aborted. Please try again."]], None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        yield history + [[message, f"An error occurred: {str(e)}"]], None

def generate_speech_sync(text, tts_model, tts_tokenizer):
    try:
        tts_input_ids = tts_tokenizer(text, return_tensors="pt").input_ids.to(device)
        tts_description = "A clear and natural voice reads the text with moderate speed and expression."
        tts_description_ids = tts_tokenizer(tts_description, return_tensors="pt").input_ids.to(device)
        
        with torch.no_grad():
            audio_generation = tts_model.generate(input_ids=tts_description_ids, prompt_input_ids=tts_input_ids)
        
        audio_buffer = audio_generation.cpu().numpy().squeeze()
        return audio_buffer if audio_buffer.size > 0 else np.array([0.0])
    except Exception as e:
        print(f"Speech generation failed: {str(e)}")
        return np.array([0.0])

@spaces.GPU(timeout=300)  # Increase timeout to 5 minutes
def process_vision_query(image, text_input):
    try:
        prompt = f"<|user|>\n<|image_1|>\n{text_input}<|end|>\n<|assistant|>\n"
        
        # Ensure the image is in the correct format
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise ValueError("Invalid image type. Expected PIL.Image.Image or numpy.ndarray")
        
        inputs = vision_processor(prompt, images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            generate_ids = vision_model.generate(
                **inputs, 
                max_new_tokens=1000, 
                eos_token_id=vision_processor.tokenizer.eos_token_id
            )
        
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = vision_processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return response
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return f"An error occurred: {str(e)}"

# Custom CSS
custom_css = """
body { background-color: #0b0f19; color: #e2e8f0; font-family: 'Arial', sans-serif;}
#custom-header { text-align: center; padding: 20px 0; background-color: #1a202c; margin-bottom: 20px; border-radius: 10px;}
#custom-header h1 { font-size: 2.5rem; margin-bottom: 0.5rem;}
#custom-header h1 .blue { color: #60a5fa;}
#custom-header h1 .pink { color: #f472b6;}
#custom-header h2 { font-size: 1.5rem; color: #94a3b8;}
.suggestions { display: flex; justify-content: center; flex-wrap: wrap; gap: 1rem; margin: 20px 0;}
.suggestion { background-color: #1e293b; border-radius: 0.5rem; padding: 1rem; display: flex; align-items: center; transition: transform 0.3s ease; width: 200px;}
.suggestion:hover { transform: translateY(-5px);}
.suggestion-icon { font-size: 1.5rem; margin-right: 1rem; background-color: #2d3748; padding: 0.5rem; border-radius: 50%;}
.gradio-container { max-width: 100% !important;}
#component-0, #component-1, #component-2 { max-width: 100% !important;}
footer { text-align: center; margin-top: 2rem; color: #64748b;}
"""

# Custom HTML for the header
custom_header = """
<div id="custom-header">
    <h1><span class="blue">Phi 3.5</span> <span class="pink">Multimodal Assistant</span></h1>
    <h2>Text and Vision AI at Your Service</h2>
</div>
"""

# Custom HTML for suggestions
custom_suggestions = """
<div class="suggestions">
    <div class="suggestion">
        <span class="suggestion-icon">💬</span>
        <p>Chat with the Text Model</p>
    </div>
    <div class="suggestion">
        <span class="suggestion-icon">🖼️</span>
        <p>Analyze Images with Vision Model</p>
    </div>
    <div class="suggestion">
        <span class="suggestion-icon">🤖</span>
        <p>Get AI-generated responses</p>
    </div>
    <div class="suggestion">
        <span class="suggestion-icon">🔍</span>
        <p>Explore advanced options</p>
    </div>
</div>
"""

# Gradio interface
with gr.Blocks(css=custom_css, theme=gr.themes.Base().set(
    body_background_fill="#0b0f19",
    body_text_color="#e2e8f0",
    button_primary_background_fill="#3b82f6",
    button_primary_background_fill_hover="#2563eb",
    button_primary_text_color="white",
    block_title_text_color="#94a3b8",
    block_label_text_color="#94a3b8",
)) as demo:
    gr.HTML(custom_header)

    with gr.Tab("Text Model (Phi-3.5-mini)"):
        chatbot = gr.Chatbot(height=400)
        msg = gr.Textbox(label="Message", placeholder="Type your message here...")
        audio_output = gr.Audio(label="Generated Speech", autoplay=True)
        with gr.Accordion("Advanced Options", open=False):
            system_prompt = gr.Textbox(value="You are a helpful assistant", label="System Prompt")
            temperature = gr.Slider(minimum=0, maximum=1, step=0.1, value=0.8, label="Temperature")
            max_new_tokens = gr.Slider(minimum=128, maximum=8192, step=1, value=1024, label="Max new tokens")
            top_p = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, value=1.0, label="top_p")
            top_k = gr.Slider(minimum=1, maximum=20, step=1, value=20, label="top_k")
            use_tts = gr.Checkbox(label="Enable Text-to-Speech", value=True)
        
        submit_btn = gr.Button("Submit", variant="primary")
        clear_btn = gr.Button("Clear Chat", variant="secondary")

        def clear_chat():
            return None

        submit_btn.click(stream_text_chat, 
                         inputs=[msg, chatbot, system_prompt, temperature, max_new_tokens, top_p, top_k, use_tts], 
                         outputs=[chatbot, audio_output])
        clear_btn.click(clear_chat, outputs=chatbot)

    with gr.Tab("Vision Model (Phi-3.5-vision)"):
        with gr.Row():
            with gr.Column(scale=1):
                vision_input_img = gr.Image(label="Upload an Image", type="pil")
                vision_text_input = gr.Textbox(label="Ask a question about the image", placeholder="What do you see in this image?")
                vision_submit_btn = gr.Button("Analyze Image", variant="primary")
            with gr.Column(scale=1):
                vision_output_text = gr.Textbox(label="AI Analysis", lines=10)
        
        vision_submit_btn.click(process_vision_query, inputs=[vision_input_img, vision_text_input], outputs=vision_output_text)

    gr.HTML("<footer>Powered by Phi 3.5 Multimodal AI</footer>")

if __name__ == "__main__":
    demo.launch(share=True)