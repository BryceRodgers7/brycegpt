"""
VoyagerGPT Backend API
FastAPI service for text generation using multiple GPT models
"""

import os
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Literal
import logging
import json
import threading
from dataclasses import dataclass

from model import GPTLanguageModel, BLOCK_SIZE

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="BryceGPT API",
    description="Multi-model GPT service for text generation",
    version="2.0.0"
)

# Add CORS middleware to allow requests from Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Streamlit app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = 'cpu'


@dataclass
class ModelConfig:
    """Configuration for a model including its vocabulary"""
    name: str
    model_path: str
    training_data_path: str
    chars: list = None
    stoi: dict = None
    itos: dict = None
    vocab_size: int = 0
    
    def __post_init__(self):
        """Load vocabulary from training data"""
        if self.chars is None:
            self._load_vocabulary()
    
    def _load_vocabulary(self):
        """Extract vocabulary from training data file"""
        logger.info(f"Loading vocabulary for {self.name} from {self.training_data_path}")
        with open(self.training_data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Extract unique characters and sort them
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        
        # Create character <-> integer mappings
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        
        logger.info(f"Loaded vocabulary for {self.name}: {self.vocab_size} characters")
    
    def encode(self, s: str) -> list:
        """Encode string to list of integers"""
        return [self.stoi[c] for c in s]
    
    def decode(self, l: list) -> str:
        """Decode list of integers to string"""
        return ''.join([self.itos[i] for i in l])


# Model configurations
MODEL_CONFIGS = {
    "voyager": ModelConfig(
        name="voyager",
        model_path=os.path.join(os.path.dirname(__file__), 'models', 'voyagerModel.pth'),
        training_data_path=os.path.join(os.path.dirname(__file__), 'training', 'voyager_dense.txt')
    ),
    "shakespeare": ModelConfig(
        name="shakespeare",
        model_path=os.path.join(os.path.dirname(__file__), 'models', 'shakespeareModel.pth'),
        training_data_path=os.path.join(os.path.dirname(__file__), 'training', 'shakespeare.txt')
    )
}


class ModelCache:
    """Thread-safe lazy-loading model cache"""
    
    def __init__(self):
        self._models = {}
        self._locks = {name: threading.Lock() for name in MODEL_CONFIGS.keys()}
        self._global_lock = threading.Lock()
    
    def get_model(self, model_name: str) -> tuple[GPTLanguageModel, ModelConfig]:
        """
        Get a model from cache or load it if not cached.
        Thread-safe: concurrent requests for the same model will only load once.
        
        Returns:
            tuple: (model, config)
        """
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")
        
        # Fast path: model already loaded
        if model_name in self._models:
            return self._models[model_name], MODEL_CONFIGS[model_name]
        
        # Slow path: need to load model (with lock to prevent duplicate loading)
        with self._locks[model_name]:
            # Double-check after acquiring lock (another thread might have loaded it)
            if model_name in self._models:
                return self._models[model_name], MODEL_CONFIGS[model_name]
            
            # Load the model
            config = MODEL_CONFIGS[model_name]
            logger.info(f"Loading model '{model_name}' from {config.model_path}")
            
            model = GPTLanguageModel(vocab_size=config.vocab_size)
            model.load_state_dict(
                torch.load(config.model_path, map_location=torch.device(device), weights_only=True)
            )
            model.eval()
            
            # Cache the model
            self._models[model_name] = model
            
            logger.info(f"Model '{model_name}' loaded successfully on {device}")
            return model, config
    
    def is_loaded(self, model_name: str) -> bool:
        """Check if a model is currently loaded in cache"""
        return model_name in self._models
    
    def get_loaded_models(self) -> list:
        """Get list of currently loaded model names"""
        return list(self._models.keys())


# Global model cache
model_cache = ModelCache()


class GenerateRequest(BaseModel):
    """Request model for text generation"""
    model: Literal["voyager", "shakespeare"] = Field(description="Model to use for generation")
    seed: int = Field(default=1337, description="Random seed for reproducibility")
    temperature: float = Field(default=0.1, ge=0.01, le=2.0, description="Temperature for sampling (0.01-2.0)")
    max_tokens: int = Field(default=100, ge=1, le=500, description="Maximum number of tokens to generate")
    context: Optional[list] = Field(default=None, description="Context tokens from previous generation (optional)")


class GenerateResponse(BaseModel):
    """Response model for text generation"""
    model: str = Field(description="Model used for generation")
    text: str = Field(description="Generated text")
    tokens: list = Field(description="Token indices of the generated text")
    generation_time: float = Field(description="Time taken to generate text in seconds")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    available_models: list
    loaded_models: list
    device: str


@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "BryceGPT API - Multi-Model Text Generation",
        "version": "2.0.0",
        "available_models": list(MODEL_CONFIGS.keys()),
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        available_models=list(MODEL_CONFIGS.keys()),
        loaded_models=model_cache.get_loaded_models(),
        device=device
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """
    Generate text using specified GPT model
    
    Args:
        request: GenerateRequest with model name, seed, temperature, max_tokens, and optional context
    
    Returns:
        GenerateResponse with generated text, tokens, and generation time
    """
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
        
        # Load model (lazy-loaded, thread-safe)
        try:
            model, config = model_cache.get_model(request.model)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
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
                context_tokens = config.encode(context_text)
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
        
        # Decode tokens to text using model-specific decoder
        text = config.decode(generated)
        
        generation_time = time.time() - start_time
        
        logger.info(f"Generated {len(generated)} tokens in {generation_time:.2f}s using model '{request.model}'")
        
        return GenerateResponse(
            model=request.model,
            text=text,
            tokens=generated,
            generation_time=generation_time
        )
    
    except HTTPException:
        raise
    except Exception as e:
        # Prepare error logging with truncated context
        error_context = {
            "model": request.model,
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


@app.get("/vocab/{model_name}", response_model=dict)
async def get_vocabulary(model_name: str):
    """Get vocabulary for a specific model"""
    if model_name not in MODEL_CONFIGS:
        raise HTTPException(
            status_code=404, 
            detail=f"Model '{model_name}' not found. Available: {list(MODEL_CONFIGS.keys())}"
        )
    
    config = MODEL_CONFIGS[model_name]
    return {
        "model": model_name,
        "vocab_size": config.vocab_size,
        "characters": config.chars,
        "block_size": BLOCK_SIZE
    }


@app.get("/models", response_model=dict)
async def list_models():
    """List all available models and their status"""
    models_info = {}
    for name, config in MODEL_CONFIGS.items():
        models_info[name] = {
            "name": name,
            "vocab_size": config.vocab_size,
            "loaded": model_cache.is_loaded(name),
            "model_path": config.model_path,
            "training_data": config.training_data_path
        }
    return {
        "models": models_info,
        "loaded_models": model_cache.get_loaded_models()
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)

