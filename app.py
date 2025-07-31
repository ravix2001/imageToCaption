# Deployed on HuggingFace

import gradio as gr
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, T5Tokenizer, T5ForConditionalGeneration

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load BLIP
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=False)
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
blip_model.eval()

# Load T5 fine-tuned model
t5_model = T5ForConditionalGeneration.from_pretrained("ravix2001/caption_hashtag_model").to(device)
t5_tokenizer = T5Tokenizer.from_pretrained("ravix2001/caption_hashtag_model")
t5_model.eval()

def generate_caption_hashtags(image):
    # Step 1: Generate description using BLIP
    inputs = blip_processor(image, return_tensors="pt").to(device)
    with torch.no_grad():
        output = blip_model.generate(**inputs)
    description = blip_processor.decode(output[0], skip_special_tokens=True)

    # Step 2: Generate caption & hashtags using T5
    t5_input_text = "generate_caption_and_hashtags: " + description
    t5_inputs = t5_tokenizer(t5_input_text, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        t5_output_ids = t5_model.generate(**t5_inputs, max_length=128)
    result = t5_tokenizer.decode(t5_output_ids[0], skip_special_tokens=True)

    return f"📄 Description:\n{description}\n\n📢 Caption & Hashtags:\n{result}"

# Gradio Interface
demo = gr.Interface(
    fn=generate_caption_hashtags,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(label="Generated Output"),
    title="📸 Caption & Hashtag Generator",
    description="Upload an image to generate a caption and relevant hashtags using BLIP + T5."
)

demo.launch(share=True)
