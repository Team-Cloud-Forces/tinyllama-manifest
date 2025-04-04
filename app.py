import os
import time
import json
import logging
import sys
import torch
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
from werkzeug.serving import WSGIRequestHandler

# Set a higher timeout for Werkzeug's request handling
WSGIRequestHandler.timeout = 300

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('tinyllama')

app = Flask(__name__)

# Configuration values
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
REQUEST_TIMEOUT = 300  # 5 minutes

# Metadata for health checks
start_time = time.time()

# Initialize TinyLlama model and tokenizer
model = None
tokenizer = None
device = None

def initialize_model():
    global model, tokenizer, device
    try:
        logger.info('=== Starting TinyLlama Initialization ===')
        logger.info('Loading model and tokenizer...')
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            padding_side='left'  # Consistent padding
        )
        
        # Ensure we have proper padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        logger.info('Model and tokenizer loaded successfully')

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        logger.info(f'Model moved to device: {device}')

        # Perform model warmup
        logger.info('=== Starting Model Warmup ===')
        sample_prompts = [
            "Hello, how are you?",
            "What is machine learning?",
            "Tell me a short story."
        ]

        for idx, prompt in enumerate(sample_prompts, 1):
            try:
                logger.info(f'Warmup iteration {idx}/{len(sample_prompts)}: "{prompt}"')
                messages = [{"role": "user", "content": prompt}]
                chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                
                inputs = tokenizer(
                    chat_prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Generate with shorter length for warmup but same parameters as production
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=50,  # Increased for visible responses
                    temperature=0.7,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
                
                # Decode and log the response
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = response.replace(chat_prompt, "").strip()
                logger.info(f'Warmup response {idx}: "{response}"')
                logger.info(f'Warmup iteration {idx} completed successfully')
            except Exception as e:
                logger.error(f'Error in warmup iteration {idx}: {str(e)}', exc_info=True)

        logger.info('=== Model Warmup Complete ===')
        return True
    except Exception as e:
        logger.error(f"Critical initialization error: {str(e)}", exc_info=True)
        return False

@app.before_request
def log_request_info():
    if request.path != '/health':
        app.logger.info('Headers: %s', request.headers)
        app.logger.info('Body: %s', request.get_data())

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    
    if not data or 'prompt' not in data:
        return jsonify({"error": "Missing prompt in request"}), 400
    
    try:
        prompt = data.get('prompt', '')
        max_length = min(data.get('max_length', 512), 1024)  # Cap at 1024
        temperature = data.get('temperature', 0.7)
        system_message = data.get('system_message', 'You are a helpful assistant, try to answer the user and help them while remaining kind and positive.')

        logger.info(f"Received generation request for prompt: {prompt[:100]}...")

        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = tokenizer(
            chat_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        logger.info("Starting generation...")
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_length,
            temperature=temperature,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        logger.info("Generation completed")
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(chat_prompt, "")
        
        logger.info(f"Generated response length: {len(response)}")
        logger.info(f"Generated response (start): {response[:100]}...")
        return jsonify({"prompt": prompt, "generated_text": response})
        
    except Exception as e:
        logger.error(f"Error in generation: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Failed to generate response",
            "details": str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model_device": str(device) if device else "not_initialized",
        "model_loaded": model is not None,
        "interface_uptime": time.time() - start_time
    })

@app.route('/model-info', methods=['GET'])
def model_information():
    return jsonify({
        "model_name": MODEL_NAME,
        "interface_type": "direct",
        "model_device": str(device) if device else "not_initialized",
        "started_at": start_time
    })

@app.after_request
def after_request(response):
    if request.path != '/health':
        app.logger.info('Response status: %s', response.status)
    return response

if __name__ == '__main__':
    # Initialize the model before starting the server
    if not initialize_model():
        logger.error("Failed to initialize TinyLlama model. Exiting.")
        sys.exit(1)
        
    # Configure threaded server
    WSGIRequestHandler.protocol_version = "HTTP/1.1"
    port = int(os.environ.get('PORT', 8000))
    app.run(
        host='0.0.0.0',
        port=port,
        threaded=True,  # Enable multi-threading
        processes=1,
        debug=os.environ.get('FLASK_ENV') == 'development'
    ) 