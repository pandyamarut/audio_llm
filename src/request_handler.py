import argparse
import asyncio
import base64
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

import aiohttp
import numpy as np
from openai import AsyncOpenAI

try:
    from vllm.assets.audio import AudioAsset
    VLLM_ASSETS_AVAILABLE = True
except ImportError:
    VLLM_ASSETS_AVAILABLE = False

async def encode_audio_from_url(url):
    """Fetch and encode audio from a URL."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            response.raise_for_status()
            content = await response.read()
            return base64.b64encode(content).decode('utf-8')



def list_available_audio_assets():
    """List available audio assets in vLLM."""
    if not VLLM_ASSETS_AVAILABLE:
        print("vLLM audio assets not available. Install vLLM to use audio assets.")
        return []
    
    # Get all attributes from AudioAsset that might be audio assets
    assets = [attr for attr in dir(AudioAsset) 
              if not attr.startswith('_') and not callable(getattr(AudioAsset, attr))]
    
    return assets



class QwenAudioRequest:
    """Represents a request to the Qwen2-Audio-7B model."""
    
    def __init__(self, 
                prompt: str, 
                audio_file: Optional[str] = None,
                audio_url: Optional[str] = None,
                audio_asset: Optional[str] = None,
                max_tokens: int = 256,
                temperature: float = 0.7):
        self.prompt = prompt
        self.audio_file = audio_file
        self.audio_url = audio_url
        self.audio_asset = audio_asset
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Results will be populated after processing
        self.success = False
        self.text = None
        self.error = None
        self.request_time = 0.0
        self.tokens_per_second = 0.0
        self.token_count = 0

async def process_request(client, request: QwenAudioRequest, pbar=None):
    """Process a single request."""
    start_time = time.time()
    
    try:
        # Prepare content
        content = [{"type": "text", "text": request.prompt}]
        
        # Handle different audio sources
        if request.audio_file:
            # Local audio file
            if not os.path.exists(request.audio_file):
                raise FileNotFoundError(f"Audio file not found: {request.audio_file}")
            
            with open(request.audio_file, "rb") as f:
                audio_base64 = base64.b64encode(f.read()).decode('utf-8')
            
            content.append({
                "type": "audio_url",
                "audio_url": {"url": f"data:audio/wav;base64,{audio_base64}"}
            })
        
        elif request.audio_url:
            # Remote audio URL (direct URL reference)
            content.append({
                "type": "audio_url",
                "audio_url": {"url": request.audio_url}
            })
        
        elif request.audio_asset:
            # vLLM audio asset
            if not VLLM_ASSETS_AVAILABLE:
                raise ImportError("vLLM audio assets not available. Install vLLM to use audio assets.")
            
            audio_url = AudioAsset(request.audio_asset).url
            content.append({
                "type": "audio_url",
                "audio_url": {"url": audio_url}
            })
        
        # Send the request
        response = await client.chat.completions.create(
            messages=[{"role": "user", "content": content}],
            model="Qwen/Qwen2-Audio-7B-Instruct",
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        # Process result
        request.success = True
        request.text = response.choices[0].message.content
        request.token_count = len(request.text.split())  # Approximate # How this is being split here?
        
        # Calculate metrics
        request.request_time = time.time() - start_time
        request.tokens_per_second = request.token_count / request.request_time if request.request_time > 0 else 0
        
    except Exception as e:
        request.success = False
        request.error = str(e)
    
    if pbar:
        pbar.update(1)
    
    return request