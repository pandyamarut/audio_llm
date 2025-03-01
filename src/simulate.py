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
from request_handler import QwenAudioRequest, encode_audio_from_url, process_request







async def simulate(
    server_url: str,
    requests: List[QwenAudioRequest],
    concurrency: int = 1,
    request_rate: float = 0.0,  # 0 means as fast as possible
    disable_tqdm: bool = False
):
    """Run a simulate with multiple requests and controlled concurrency."""
    # Initialize client
    client = AsyncOpenAI(api_key="EMPTY", base_url=server_url)
    
    # Set up progress bar
    pbar = None if disable_tqdm else tqdm(total=len(requests))
    
    # Create semaphore to limit concurrency
    semaphore = asyncio.Semaphore(concurrency)
    
    async def limited_process(request):
        async with semaphore:
            return await process_request(client, request, pbar)
    
    # Create and schedule tasks
    tasks = []
    simulate_start_time = time.time()
    
    if request_rate > 0:
        # Add requests according to specified rate
        for i, request in enumerate(requests):
            # Delay to match request rate (requests per second)
            delay = i / request_rate
            await asyncio.sleep(max(0, simulate_start_time + delay - time.time()))
            tasks.append(asyncio.create_task(limited_process(request)))
    else:
        # Add all requests at once (as fast as possible, limited by semaphore)
        tasks = [asyncio.create_task(limited_process(request)) for request in requests]
    
    # Wait for all tasks to complete
    completed_requests = await asyncio.gather(*tasks)
    
    if pbar:
        pbar.close()
    
    simulate_duration = time.time() - simulate_start_time
    
    return completed_requests, simulate_duration