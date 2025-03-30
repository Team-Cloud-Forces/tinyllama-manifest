import os
import time
import json
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

# Configuration for the TinyLlama service
TINYLLAMA_SERVICE_HOST = os.environ.get("TINYLLAMA_SERVICE_HOST", "tinyllama-service")
TINYLLAMA_SERVICE_PORT = os.environ.get("TINYLLAMA_SERVICE_PORT", "8000")
TINYLLAMA_SERVICE_URL = f"http://{TINYLLAMA_SERVICE_HOST}:{TINYLLAMA_SERVICE_PORT}"

# Metadata for health checks
start_time = time.time()
model_info = {
    "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "interface_type": "proxy",
    "backend_service": TINYLLAMA_SERVICE_URL,
    "started_at": start_time,
}

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    
    if not data or 'prompt' not in data:
        return jsonify({"error": "Missing prompt in request"}), 400
    
    try:
        # Forward the request to the TinyLlama service
        response = requests.post(
            f"{TINYLLAMA_SERVICE_URL}/generate",
            json=data,
            timeout=30
        )
        response.raise_for_status()
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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=os.environ.get('FLASK_ENV') == 'development') 