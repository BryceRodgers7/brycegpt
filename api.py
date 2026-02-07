"""
VoyagerGPT Backend API
FastAPI service for text generation using the VoyagerGPT model
"""

import os
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import logging
import json

from model import GPTLanguageModel, encode, decode, BLOCK_SIZE

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="VoyagerGPT API",
    description="A bigram GPT built from scratch for Star Trek text generation",
    version="1.0.0"
)

# Add CORS middleware to allow requests from Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Streamlit app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None
device = 'cpu'

# Model path
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'voyagerModel.pth')


class GenerateRequest(BaseModel):
    """Request model for text generation"""
    seed: int = Field(default=1337, description="Random seed for reproducibility")
    temperature: float = Field(default=0.1, ge=0.01, le=2.0, description="Temperature for sampling (0.01-2.0)")
    max_tokens: int = Field(default=100, ge=1, le=500, description="Maximum number of tokens to generate")
    context: Optional[list] = Field(default=None, description="Context tokens from previous generation (optional)")


class GenerateResponse(BaseModel):
    """Response model for text generation"""
    text: str = Field(description="Generated text")
    tokens: list = Field(description="Token indices of the generated text")
    generation_time: float = Field(description="Time taken to generate text in seconds")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    device: str


@app.on_event("startup")
async def load_model():
    """Load the model on startup"""
    global model
    try:
        logger.info(f"Loading model from {MODEL_PATH}")
        model = GPTLanguageModel()
        model.load_state_dict(
            torch.load(MODEL_PATH, map_location=torch.device(device), weights_only=True)
        )
        model.eval()
        logger.info(f"Model loaded successfully on {device}")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise


@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "VoyagerGPT API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        device=device
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """
    Generate text using VoyagerGPT
    
    Args:
        request: GenerateRequest with seed, temperature, max_tokens, and optional context
    
    Returns:
        GenerateResponse with generated text, tokens, and generation time
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        import time
        start_time = time.time()
        
        # Log the incoming request with truncated context for readability
        request_dict = request.model_dump()
        log_dict = request_dict.copy()
        if request.context and len(request.context) > 20:
            log_dict['context'] = f"[{len(request.context)} tokens - showing first 10 and last 10]"
            log_dict['context_preview'] = {
                'first': request.context[:10],
                'last': request.context[-10:]
            }
        logger.info(f"Generation request received: {json.dumps(log_dict)}")
        
        # Set random seed
        torch.manual_seed(request.seed)
        
        # Prepare context
        if request.context and len(request.context) > 0:
            # Check if context contains strings (characters) or integers (token IDs)
            context_tokens = request.context
            
            # If context contains strings, we need to encode them to token IDs
            if isinstance(context_tokens[0], str):
                logger.info(f"Context contains strings, encoding to token IDs")
                # Join the characters back into a string and encode
                context_text = ''.join(context_tokens)
                context_tokens = encode(context_text)
                logger.info(f"Encoded {len(context_text)} characters to {len(context_tokens)} tokens")
            
            # Truncate context to BLOCK_SIZE if necessary (model can only handle BLOCK_SIZE tokens)
            if len(context_tokens) > BLOCK_SIZE:
                logger.warning(f"Context length ({len(context_tokens)}) exceeds BLOCK_SIZE ({BLOCK_SIZE}). Truncating to last {BLOCK_SIZE} tokens.")
                context_tokens = context_tokens[-BLOCK_SIZE:]
            
            # Use provided context (already a list, so wrap it as a batch of 1)
            context = torch.tensor(context_tokens, dtype=torch.long, device=device).unsqueeze(0)
            logger.info(f"Using context with {len(context_tokens)} tokens")
        else:
            # Start with zero token
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
            logger.info("Starting with zero token context")
        
        # Generate text
        with torch.no_grad():
            generated = model.generate(
                context, 
                max_new_tokens=request.max_tokens,
                temperature=request.temperature
            )[0].tolist()
        
        # Decode tokens to text
        text = decode(generated)
        
        generation_time = time.time() - start_time
        
        logger.info(f"Generated {len(generated)} tokens in {generation_time:.2f}s")
        
        return GenerateResponse(
            text=text,
            tokens=generated,
            generation_time=generation_time
        )
    
    except Exception as e:
        # Prepare error logging with truncated context
        error_context = {
            "seed": request.seed,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "context_length": len(request.context) if request.context else 0
        }
        if request.context and len(request.context) > 0:
            if len(request.context) <= 20:
                error_context["context"] = request.context
            else:
                error_context["context_preview"] = {
                    "first_10": request.context[:10],
                    "last_10": request.context[-10:],
                    "total_length": len(request.context)
                }
        
        logger.error(
            f"Generation failed: {str(e)}", 
            exc_info=True,
            extra={"request_info": json.dumps(error_context)}
        )
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.get("/vocab", response_model=dict)
async def get_vocabulary():
    """Get the model's vocabulary"""
    from model import CHARS, VOCAB_SIZE
    return {
        "vocab_size": VOCAB_SIZE,
        "characters": CHARS,
        "block_size": BLOCK_SIZE
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)

