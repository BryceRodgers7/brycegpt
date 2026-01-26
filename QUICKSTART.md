# VoyagerGPT Backend - Quick Start Guide

Get up and running with the VoyagerGPT backend in minutes!

## 🚀 Quick Local Testing

### Option 1: Python (No Docker)

```bash
# Navigate to the brycegpt directory
cd brycegpt

# Install dependencies
pip install -r requirements.txt

# Run the API
python api.py
```

The API will be available at `http://localhost:8080`

### Option 2: Docker

```bash
# Navigate to the brycegpt directory
cd brycegpt

# Build the Docker image
docker build -t brycegpt .

# Run the container
docker run -p 8080:8080 brycegpt
```

The API will be available at `http://localhost:8080`

## 🧪 Test the API

### 1. Check Health

```bash
curl http://localhost:8080/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu"
}
```

### 2. Generate Text

```bash
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{
    "seed": 1337,
    "temperature": 0.1,
    "max_tokens": 100
  }'
```

Expected response:
```json
{
  "text": "Generated Star Trek text...",
  "tokens": [1, 2, 3, ...],
  "generation_time": 5.23
}
```

### 3. View Interactive Docs

Open your browser to:
- Swagger UI: `http://localhost:8080/docs`
- ReDoc: `http://localhost:8080/redoc`

## 🌐 Deploy to Google Cloud Run

### Using the Deployment Script (Recommended)

**Windows:**
```bash
cd brycegpt
deploy.bat
```

**Linux/Mac:**
```bash
cd brycegpt
chmod +x deploy.sh
./deploy.sh
```

### Manual Deployment

```bash
# Set your project
export PROJECT_ID=your-project-id
gcloud config set project $PROJECT_ID

# Build and deploy
gcloud builds submit --tag gcr.io/$PROJECT_ID/brycegpt

gcloud run deploy brycegpt \
  --image gcr.io/$PROJECT_ID/brycegpt \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300
```

## 🔗 Connect Your Frontend

After deployment, update your Streamlit frontend:

```python
import streamlit as st
import requests

# Replace with your Cloud Run URL
API_URL = "https://brycegpt-xxxxx-uc.a.run.app"

# Generate text
response = requests.post(
    f"{API_URL}/generate",
    json={
        "seed": 1337,
        "temperature": 0.1,
        "max_tokens": 100
    }
)

result = response.json()
st.write(result["text"])
```

See `frontend_example.py` for a complete working example.

## 📊 Test with Python

```python
import requests

API_URL = "http://localhost:8080"  # or your Cloud Run URL

# Generate text
response = requests.post(
    f"{API_URL}/generate",
    json={
        "seed": 1337,
        "temperature": 0.5,
        "max_tokens": 200,
        "context": None  # or provide previous tokens for continuation
    }
)

if response.status_code == 200:
    result = response.json()
    print(f"Generated text:\n{result['text']}")
    print(f"\nTime: {result['generation_time']:.2f}s")
    print(f"Tokens: {len(result['tokens'])}")
else:
    print(f"Error: {response.status_code}")
```

## 🎯 Generation Tips

### Temperature Settings

- **0.1 - 0.3**: Very coherent, predictable (good for structured text)
- **0.4 - 0.7**: Balanced creativity and coherence
- **0.8 - 1.0**: Very creative, more random (experimental)

### Context Continuation

To continue a story, save the `tokens` from the response and pass them as `context` in the next request:

```python
# First generation
response1 = requests.post(f"{API_URL}/generate", json={
    "seed": 1337,
    "temperature": 0.5,
    "max_tokens": 100
})
tokens = response1.json()["tokens"]

# Continue the story
response2 = requests.post(f"{API_URL}/generate", json={
    "seed": 1337,
    "temperature": 0.5,
    "max_tokens": 100,
    "context": tokens  # Continue from previous generation
})
```

## 🐛 Troubleshooting

### "Model not loaded" error
- Ensure `voyagerModel.pth` is in the `brycegpt` directory
- Check that the model file is not corrupted

### Port already in use
```bash
# Find and kill the process using port 8080
# Windows:
netstat -ano | findstr :8080
taskkill /PID <PID> /F

# Linux/Mac:
lsof -ti:8080 | xargs kill -9
```

### Slow generation times
- This is normal on CPU (10-30 seconds for 100 tokens)
- GPU acceleration would be much faster but isn't available on Cloud Run
- Consider reducing `max_tokens` for faster responses

### Docker build fails
- Ensure you're in the `brycegpt` directory
- Check that all required files are present
- Verify Docker is running

## 📖 Next Steps

- See `README.md` for detailed deployment options
- Check `frontend_example.py` for frontend integration
- Visit `/docs` endpoint for full API documentation
- Monitor costs in Google Cloud Console

## 💡 Pro Tips

1. **Use Docker for consistency** - Ensures same environment locally and in production
2. **Test locally first** - Always test changes before deploying to Cloud Run
3. **Monitor costs** - Set up billing alerts in Google Cloud Console
4. **Use min-instances** - For better performance (but costs more)
5. **Version your deployments** - Use tags for different versions

Happy generating! 🚀

