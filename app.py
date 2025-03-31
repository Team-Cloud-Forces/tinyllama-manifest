import os
import time
import json
import requests
from flask import Flask, request, jsonify
import logging
from werkzeug.serving import WSGIRequestHandler

app = Flask(__name__)

# Configuration for the TinyLlama service
TINYLLAMA_SERVICE_HOST = os.environ.get("TINYLLAMA_SERVICE_HOST", "tinyllama-service")
TINYLLAMA_SERVICE_PORT = os.environ.get("TINYLLAMA_SERVICE_PORT", "8000")
TINYLLAMA_SERVICE_URL = f"http://{TINYLLAMA_SERVICE_HOST}:{TINYLLAMA_SERVICE_PORT}"

# Request timeout configuration
REQUEST_TIMEOUT = 300  # 5 minutes, matching backend timeout

# Metadata for health checks
start_time = time.time()
model_info = {
    "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "interface_type": "proxy",
    "backend_service": TINYLLAMA_SERVICE_URL,
    "started_at": start_time,
}

# Set up logging
logging.basicConfig(level=logging.INFO)

@app.before_request
def log_request_info():
    app.logger.info('Headers: %s', request.headers)
    app.logger.info('Body: %s', request.get_data())

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    
    if not data or 'prompt' not in data:
        return jsonify({"error": "Missing prompt in request"}), 400
    
    try:
        # Forward the request to the TinyLlama service with increased timeout
        response = requests.post(
            f"{TINYLLAMA_SERVICE_URL}/generate",
            json=data,
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        app.logger.info('Request processed successfully with status code: %s', response.status_code)
        return jsonify(response.json())
        
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Error communicating with TinyLlama service: {str(e)}")
        return jsonify({
            "error": "Failed to communicate with TinyLlama backend service",
            "details": str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "interface_uptime": time.time() - start_time
    })

@app.route('/model-info', methods=['GET'])
def model_information():
    try:
        # Try to get model info from the backend
        response = requests.get(f"{TINYLLAMA_SERVICE_URL}/model-info", timeout=5)
        if response.status_code == 200:
            return jsonify(response.json())
    except requests.exceptions.RequestException:
        pass
    
    # Fallback model info if backend doesn't support this endpoint
    return jsonify({
        "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "interface_type": "proxy",
        "backend_service": TINYLLAMA_SERVICE_URL
    })

@app.after_request
def after_request(response):
    app.logger.info('Response status: %s', response.status)
    return response

if __name__ == '__main__':
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