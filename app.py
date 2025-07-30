from flask import Flask, request, jsonify
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import os

app = Flask(__name__)

# Load BLIP model and processor (image -> description)
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=False) # use_fast=False for low memory usage in deployment
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda" if torch.cuda.is_available() else "cpu")
blip_model.eval()

# Load fine-tuned T5 model and tokenizer (description -> caption + hashtags)
t5_model = T5ForConditionalGeneration.from_pretrained("ravix2001/caption_hashtag_model")
t5_tokenizer = T5Tokenizer.from_pretrained("ravix2001/caption_hashtag_model")
t5_model.eval()

@app.route('/')
def home():
    return "Image to Caption & Hashtag Generator API is running."

@app.route('/generate', methods=['POST'])
def generate():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded'}), 400

    # Step 1: Load image
    image_file = request.files['image']
    image = Image.open(image_file).convert('RGB')

    # Step 2: Generate description using BLIP
    inputs = blip_processor(image, return_tensors="pt")
    with torch.no_grad():
        output = blip_model.generate(**inputs)
    description = blip_processor.decode(output[0], skip_special_tokens=True)

    # Step 3: Generate caption & hashtags using fine-tuned T5
    t5_input_text = "generate_caption_and_hashtags: " + description
    t5_inputs = t5_tokenizer(t5_input_text, return_tensors="pt", padding=True)

    with torch.no_grad():
        t5_output_ids = t5_model.generate(**t5_inputs, max_length=128)
    final_result = t5_tokenizer.decode(t5_output_ids[0], skip_special_tokens=True)

    return jsonify({
        'description': description,
        'caption_hashtags': final_result
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use PORT env var from Render
    app.run(host="0.0.0.0", port=port)
