#!/usr/bin/env python3
"""
Versatile high-concurrency client for Qwen2-Audio-7B supporting multiple audio input methods.
"""
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
from request_handler import QwenAudioRequest, process_request, list_available_audio_assets
from simulate import simulate
from utils import calculate_metrics
# Try to import vLLM's audio assets (optional)

def calculate_metrics(requests: List[QwenAudioRequest], duration: float) -> Dict:
    """Calculate and return simulate metrics."""
    # Filter successful requests
    successful = [r for r in requests if r.success]
    
    # Calculate metrics
    metrics = {
        "duration": duration,
        "completed": len(successful),
        "failed": len(requests) - len(successful),
        "total_requests": len(requests),
        "success_rate": len(successful) / len(requests) if requests else 0,
        "request_throughput": len(successful) / duration if duration > 0 else 0,
        "total_tokens": sum(r.token_count for r in successful),
        "avg_tokens_per_request": sum(r.token_count for r in successful) / len(successful) if successful else 0,
        "token_throughput": sum(r.token_count for r in successful) / duration if duration > 0 else 0,
        "avg_request_time": sum(r.request_time for r in successful) / len(successful) if successful else 0,
        "avg_tokens_per_second": sum(r.tokens_per_second for r in successful) / len(successful) if successful else 0,
        "max_tokens_per_second": max((r.tokens_per_second for r in successful), default=0),
        "min_tokens_per_second": min((r.tokens_per_second for r in successful), default=0),
    }
    
    # Add percentiles for tokens/sec
    if successful:
        tokens_per_second = [r.tokens_per_second for r in successful]
        for p in [50, 90, 95, 99]:
            metrics[f"p{p}_tokens_per_second"] = np.percentile(tokens_per_second, p)
    
    return metrics



async def main():
    parser = argparse.ArgumentParser(description="Qwen2-Audio-7B Advanced Client")
    
    # Core parameters
    parser.add_argument("--server", default="http://localhost:8000/v1", help="Server URL")
    parser.add_argument("--prompt", default="Describe what you hear in this audio.", help="Text prompt")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max tokens to generate")
    parser.add_argument("--output", help="Output file")
    
    # Audio sources (mutually exclusive)
    audio_group = parser.add_mutually_exclusive_group()
    audio_group.add_argument("--audio-file", help="Path to local audio file")
    audio_group.add_argument("--audio-url", help="URL to remote audio file")
    audio_group.add_argument("--audio-asset", help="vLLM audio asset name (e.g., 'winning_call')")
    
    # simulate options
    parser.add_argument("--simulate", action="store_true", help="Run a simulation")
    parser.add_argument("--concurrency", type=int, default=10, help="Maximum concurrency")
    parser.add_argument("--requests", type=int, default=100, help="Total number of requests")
    parser.add_argument("--rate", type=float, default=0, help="Request rate (0 = as fast as possible)")
    parser.add_argument("--silent", action="store_true", help="Disable progress bar")
    
    # Utility commands
    parser.add_argument("--list-assets", action="store_true", help="List available vLLM audio assets")
    
    args = parser.parse_args()
    
    if args.list_assets:
        assets = list_available_audio_assets()
        if assets:
            print("Available vLLM audio assets:")
            for asset in assets:
                print(f" - {asset}")
        return
    
    # Determine if we're simulateing or processing a single request
    if args.simulate:
        # Run simulate
        requests = []
        for i in range(args.requests):
            request = QwenAudioRequest(
                prompt=args.prompt,
                audio_file=args.audio_file,
                audio_url=args.audio_url,
                audio_asset=args.audio_asset,
                max_tokens=args.max_tokens,
                temperature=0.7
            )
            requests.append(request)
        
        print(f"Starting simulate with {len(requests)} requests")
        print(f"Concurrency: {args.concurrency}")
        print(f"Request rate: {args.rate if args.rate > 0 else 'unlimited'} req/s")
        
        if args.audio_file:
            print(f"Using audio file: {args.audio_file}")
        elif args.audio_url:
            print(f"Using audio URL: {args.audio_url}")
        elif args.audio_asset:
            print(f"Using audio asset: {args.audio_asset}")
        
        # Run the simulate
        results, duration = await simulate(
            server_url=args.server,
            requests=requests,
            concurrency=args.concurrency,
            request_rate=args.rate,
            disable_tqdm=args.silent
        )
        
        # Calculate metrics
        metrics = calculate_metrics(results, duration)
        
        # Display simulation results
        print("\n" + "=" * 60)
        print("Simulation RESULTS")
        print("=" * 60)
        print(f"Total requests:       {metrics['total_requests']}")
        print(f"Successful requests:  {metrics['completed']}")
        print(f"Failed requests:      {metrics['failed']}")
        print(f"Success rate:         {metrics['success_rate']:.2%}")
        print(f"Duration:             {metrics['duration']:.2f}s")
        print(f"Request throughput:   {metrics['request_throughput']:.2f} req/s")
        print(f"Token throughput:     {metrics['token_throughput']:.2f} tokens/s")
        print(f"Avg tokens/request:   {metrics['avg_tokens_per_request']:.2f}")
        print(f"Avg request time:     {metrics['avg_request_time']:.2f}s")
        print(f"Avg tokens/second:    {metrics['avg_tokens_per_second']:.2f}")
        print(f"Min tokens/second:    {metrics['min_tokens_per_second']:.2f}")
        print(f"Max tokens/second:    {metrics['max_tokens_per_second']:.2f}")
# Print percentile metrics (P50, P90, P99) if available:
        print(f"P50 tokens/second:    {metrics.get('p50_tokens_per_second', 0):.2f}")
        print(f"P90 tokens/second:    {metrics.get('p90_tokens_per_second', 0):.2f}")
        print(f"P99 tokens/second:    {metrics.get('p99_tokens_per_second', 0):.2f}")

        
        # Save simulate results if requested
        if args.output:
            # For simulates, save the metrics to the output file (JSON)
            metrics_file = args.output
            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"simulate metrics saved to {metrics_file}")
            
            # Save all successful outputs to individual files
            successful = [r for r in results if r.success]
            if successful:
                # Create output directory based on the output file name
                output_dir = os.path.dirname(args.output) or "."
                base_name = os.path.splitext(os.path.basename(args.output))[0]
                results_dir = os.path.join(output_dir, f"{base_name}_results")
                os.makedirs(results_dir, exist_ok=True)
                
                # Save each successful result
                for i, result in enumerate(successful):
                    output_file = os.path.join(results_dir, f"output_{i+1}.txt")
                    with open(output_file, "w") as f:
                        f.write(result.text)
                
                print(f"All {len(successful)} outputs saved to {results_dir}/")
    else:
        # Process a single request
        client = AsyncOpenAI(api_key="EMPTY", base_url=args.server)
        
        request = QwenAudioRequest(
            prompt=args.prompt,
            audio_file=args.audio_file,
            audio_url=args.audio_url,
            audio_asset=args.audio_asset,
            max_tokens=args.max_tokens
        )
        
        result = await process_request(client, request)
        
        if result.success:
            print("\n" + "="*60)
            print(result.text)
            print("="*60)
            print(f"Time: {result.request_time:.2f}s, Generated {result.token_count} tokens")
            print(f"Speed: {result.tokens_per_second:.2f} tokens/sec")
            
            if args.output:
                with open(args.output, "w") as f:
                    f.write(result.text)
                print(f"Output saved to {args.output}")
        else:
            print(f"Error: {result.error}")

if __name__ == "__main__":
    asyncio.run(main())