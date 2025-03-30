# TinyLlama Flask Interface

## Overview
This project provides a Flask-based web interface for an existing TinyLlama language model deployment. It acts as a proxy between clients and your deployed TinyLlama model, providing a clean API for text generation.

## Features
- Flask-based RESTful API interface to TinyLlama model
- Kubernetes deployment configuration
- Docker containerization
- Lightweight proxy design
- Health check and model information endpoints

## Prerequisites
- Python 3.9+
- Docker
- Kubernetes cluster with existing TinyLlama deployment
- kubectl

## Installation

### Clone the Repository
```bash
git clone https://github.com/your-username/tinyllama-interface.git
cd tinyllama-interface
```

### Create Virtual Environment (Recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Local Development

### Configuration
Set the following environment variables to point to your existing TinyLlama service:
```bash
export TINYLLAMA_SERVICE_HOST=your-tinyllama-service-host
export TINYLLAMA_SERVICE_PORT=8000
```

### Run the Application
```bash
python app.py
```
The interface will start on `http://localhost:8000`

## Docker Deployment

### Build Docker Image
```bash
docker build -t your-username/tinyllama-interface:latest .
```

### Push to Container Registry
```bash
docker push your-username/tinyllama-interface:latest
```

## Kubernetes Deployment

### Apply Kubernetes Manifest
Before deploying, update the following in `tinyllama-interface-deployment.yaml`:
1. Replace `${DOCKER_USERNAME}` with your Docker Hub username.
2. Ensure `TINYLLAMA_SERVICE_HOST` and `TINYLLAMA_SERVICE_PORT` match your existing TinyLlama service.

```bash
kubectl apply -f tinyllama-interface-deployment.yaml
```

## API Endpoints

### Generate Text
- **Endpoint**: `/generate`
- **Method**: POST
- **Request Body**:
  ```json
  {
    "prompt": "Your input text",
    "max_length": 100,      // Optional, default 50
    "temperature": 0.7,     // Optional, default 1.0
    "top_k": 50,            // Optional
    "top_p": 0.95           // Optional
  }
  ```
- **Response**:
  ```json
  {
    "prompt": "Original prompt",
    "generated_text": "Generated model output"
  }
  ```

### Health Check
- **Endpoint**: `/health`
- **Method**: GET
- **Response**: Returns interface and backend service status

### Model Information
- **Endpoint**: `/model-info`
- **Method**: GET
- **Response**: Provides details about the model from the backend service

## Example Curl Request
```bash
curl -X POST http://localhost:8000/generate \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Tell me a short story", "max_length": 100}'
```

## Configuration

### Environment Variables
- `FLASK_ENV`: Set to `development` or `production`
- `TINYLLAMA_SERVICE_HOST`: Hostname of the TinyLlama backend service
- `TINYLLAMA_SERVICE_PORT`: Port of the TinyLlama backend service

## Troubleshooting
- Check that the backend TinyLlama service is running and accessible
- Verify network connectivity between the interface and backend service
- Check for correct environment variable configuration

## License
Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgments
- [TinyLlama Project](https://github.com/TinyLlama/TinyLlama)
- [Flask](https://flask.palletsprojects.com/) 