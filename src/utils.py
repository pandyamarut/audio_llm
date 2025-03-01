from typing import List, Dict, Tuple
from request_handler import QwenAudioRequest
import numpy as np


def calculate_metrics(requests: List[QwenAudioRequest], duration: float) -> Tuple[Dict, List[int]]:
    """
    Calculate and return simulation metrics based on QwenAudioRequest objects.
    
    This version is adapted from the provided function and uses:
      - token_count as the output token count.
      - request.request_time as the overall latency.
      - tokens_per_second already computed per request.
    
    Since we don't have separate TTFT, ITL, or detailed latency fields, we approximate:
      - TTFT: assumed to be half of the total request time.
      - E2EL: the full request time.
    
    Args:
        requests (List[QwenAudioRequest]): List of request objects.
        duration (float): Total simulation duration in seconds.
    
    Returns:
        Tuple[Dict, List[int]]: A tuple containing:
          - metrics (Dict): A dictionary with aggregated benchmark metrics.
          - actual_output_lens (List[int]): A list of output token counts per request.
    """
    actual_output_lens: List[int] = []
    total_input = 0  # Not provided; set to 0.
    completed = 0
    good_completed = 0  # Not used since no SLO config is provided.
    
    # We'll approximate ttft as half of the request time and e2el as the full request time.
    ttfts: List[float] = []
    e2els: List[float] = []
    
    for req in requests:
        if req.success:
            # Use token_count as output length.
            output_len = req.token_count
            actual_output_lens.append(output_len)
            completed += 1
            
            # Approximate time-to-first-token (TTFT) as half of the request time.
            ttft = req.request_time / 2  
            ttfts.append(ttft)
            # End-to-end latency (E2EL) is the full request time.
            e2els.append(req.request_time)
        else:
            actual_output_lens.append(0)
    
    total_tokens = sum(actual_output_lens)
    avg_tokens_per_request = total_tokens / completed if completed else 0
    token_throughput = total_tokens / duration if duration > 0 else 0
    avg_request_time = (sum(req.request_time for req in requests if req.success) / completed
                        if completed else 0)
    tokens_per_second_list = [req.tokens_per_second for req in requests if req.success]
    avg_tokens_per_second = (sum(tokens_per_second_list) / len(tokens_per_second_list)
                             if tokens_per_second_list else 0)
    max_tokens_per_second = max(tokens_per_second_list, default=0)
    min_tokens_per_second = min(tokens_per_second_list, default=0)
    
    metrics = {
        "duration": duration,
        "completed": completed,
        "failed": len(requests) - completed,
        "total_requests": len(requests),
        "success_rate": completed / len(requests) if requests else 0,
        "request_throughput": completed / duration if duration > 0 else 0,
        "total_tokens": total_tokens,
        "avg_tokens_per_request": avg_tokens_per_request,
        "token_throughput": token_throughput,
        "avg_request_time": avg_request_time,
        "avg_tokens_per_second": avg_tokens_per_second,
        "max_tokens_per_second": max_tokens_per_second,
        "min_tokens_per_second": min_tokens_per_second,
        # Approximate TTFT metrics (in ms)
        "mean_ttft_ms": np.mean(ttfts or 0) * 1000,
        "std_ttft_ms": np.std(ttfts or 0) * 1000,
        "median_ttft_ms": np.median(ttfts or 0) * 1000,
        # E2EL metrics (in ms)
        "mean_e2el_ms": np.mean(e2els or 0) * 1000,
        "std_e2el_ms": np.std(e2els or 0) * 1000,
        "median_e2el_ms": np.median(e2els or 0) * 1000,
    }
    
    # Add percentiles for tokens/sec
    if tokens_per_second_list:
        for p in [50, 90, 95, 99]:
            metrics[f"p{p}_tokens_per_second"] = np.percentile(tokens_per_second_list, p)
    
    return metrics, actual_output_lens
