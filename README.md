# VoyagerGPT Backend API

A FastAPI backend service for the VoyagerGPT model - a bigram GPT built from scratch for Star Trek text generation.

## 📁 Project Structure

```
brycegpt/
├── api.py              # FastAPI application
├── model.py            # GPT model architecture
├── voyagerModel.pth    # Pre-trained model weights
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container configuration
├── .dockerignore       # Docker ignore patterns
└── README.md           # This file
```

## 🚀 Deployment to Google Cloud Run

### Prerequisites

1. Install [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)
2. Authenticate with Google Cloud:
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```

### Deployment Steps

#### 1. Build and Deploy using Cloud Build

From the `brycegpt` directory:

```bash
# Set your project ID
export PROJECT_ID=your-project-id

# Build the container image
gcloud builds submit --tag gcr.io/$PROJECT_ID/brycegpt

# Deploy to Cloud Run
gcloud run deploy brycegpt \
  --image gcr.io/$PROJECT_ID/brycegpt \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300
```

#### 2. Alternative: Build Locally and Push

```bash
# Build the Docker image locally
docker build -t gcr.io/$PROJECT_ID/brycegpt .

# Push to Google Container Registry
docker push gcr.io/$PROJECT_ID/brycegpt

# Deploy to Cloud Run
gcloud run deploy brycegpt \
  --image gcr.io/$PROJECT_ID/brycegpt \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300
```

### Configuration Options

- `--memory`: Increase if needed (model has 10M+ parameters)
- `--cpu`: Adjust based on performance requirements
- `--timeout`: Maximum request duration (default 300s)
- `--min-instances`: Keep warm instances (costs more but faster response)
- `--max-instances`: Limit concurrent instances

## 🧪 Local Testing

### Run with Docker

```bash
# Build the image
docker build -t brycegpt .

# Run the container
docker run -p 8080:8080 brycegpt
```

### Run without Docker

```bash
# Install dependencies
pip install -r requirements.txt

# Run the API
python api.py
```

Access the API at `http://localhost:8080`

## 📚 API Endpoints

### `GET /`
Root endpoint with API information

### `GET /health`
Health check endpoint
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu"
}
```

### `POST /generate`
Generate text using VoyagerGPT

**Request Body:**
```json
{
  "seed": 1337,
  "temperature": 0.1,
  "max_tokens": 100,
  "context": null
}
```

**Response:**
```json
{
  "text": "Generated Star Trek text...",
  "tokens": [1, 2, 3, ...],
  "generation_time": 5.23
}
```

**Parameters:**
- `seed` (int): Random seed for reproducibility (default: 1337)
- `temperature` (float): Sampling temperature 0.01-2.0 (default: 0.1)
- `max_tokens` (int): Maximum tokens to generate 1-500 (default: 100)
- `context` (list, optional): Previous generation tokens for continuation

### `GET /vocab`
Get model vocabulary information

## 💻 Frontend Integration

### Python/Streamlit Example

```python
import streamlit as st
import requests

# Your deployed Cloud Run URL
API_URL = "https://brycegpt-xxxx.run.app"

st.title("VoyagerGPT")

seed = st.sidebar.number_input("Seed", value=1337)
temperature = st.sidebar.slider("Temperature", 0.01, 1.0, 0.1)

if st.button("Generate"):
    with st.spinner("Generating..."):
        response = requests.post(
            f"{API_URL}/generate",
            json={
                "seed": seed,
                "temperature": temperature,
                "max_tokens": 100
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            st.write(result["text"])
            st.caption(f"Generated in {result['generation_time']:.2f}s")
        else:
            st.error("Generation failed")
```

### JavaScript Example

```javascript
const API_URL = 'https://brycegpt-xxxx.run.app';

async function generateText(seed, temperature, maxTokens) {
  const response = await fetch(`${API_URL}/generate`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      seed: seed,
      temperature: temperature,
      max_tokens: maxTokens
    })
  });
  
  return await response.json();
}
```

## 📊 Model Information

- **Architecture**: Transformer-based GPT
- **Parameters**: 10M+
- **Vocabulary Size**: 87 characters
- **Context Length**: 256 tokens
- **Training Data**: Star Trek scripts
- **Embedding Dimension**: 384
- **Attention Heads**: 6
- **Layers**: 6

## 🔧 Troubleshooting

### Memory Issues
If you encounter memory errors, increase the Cloud Run memory:
```bash
gcloud run services update brycegpt --memory 4Gi
```

### Timeout Issues
Increase the timeout for longer generations:
```bash
gcloud run services update brycegpt --timeout 600
```

### Cold Start Performance
To reduce cold starts, set minimum instances:
```bash
gcloud run services update brycegpt --min-instances 1
```
Note: This will incur costs even when idle.

## 💰 Cost Considerations

- **Cloud Run**: Pay per request + CPU/memory usage
- **Container Registry**: Storage costs for Docker images
- **Idle instances**: Only charged if using `--min-instances`

Monitor costs in Google Cloud Console under Billing.

## 📝 Interactive API Documentation

Once deployed, visit:
- `/docs` - Swagger UI documentation
- `/redoc` - ReDoc documentation

## 🔒 Security Considerations

For production:

1. **Remove `allow-unauthenticated`** and implement authentication
2. **Add rate limiting** to prevent abuse
3. **Set CORS origins** to your frontend domain only (in `api.py`)
4. **Enable Cloud Armor** for DDoS protection
5. **Use Secret Manager** for sensitive configuration

## 📄 License

Same license as the parent project.

