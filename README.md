# Qwen2-Audio-7B End-to-End Pipeline

This repository contains an end-to-end pipeline for running the Qwen2-Audio-7B audio language model, which can process audio and text inputs in real-time with high throughput.

## Features

- Simple client that sends audio or audio+text requests to the vLLM server
- High-performance inference supporting 100+ parallel requests
- Support for local audio files, remote URLs, and built-in audio assets
- Comprehensive benchmarking capabilities

## Performance Notes

Different server configurations yield different performance results:

```bash
# Basic configuration (~30 tokens/request)
vllm serve Qwen/Qwen2-Audio-7B-Instruct \
    --trust-remote-code \
    --max-model-len 4096 \
    --limit-mm-per-prompt audio=1 \
    --gpu-memory-utilization 0.9 \
    --max-num-batched-tokens 4096
```

```bash
# Enhanced configuration (~50 tokens/second)
vllm serve Qwen/Qwen2-Audio-7B-Instruct \
    --trust-remote-code \
    --max-model-len 4096 \
    --limit-mm-per-prompt audio=1 \
    --gpu-memory-utilization 0.9 \
    --max-num-batched-tokens 4096 \
    --dtype float16 \
    --kv-cache-dtype fp8
```

## Quick Start

```bash
# Navigate to the pipeline directory
1. Clone this repository. 
2. cd audio_llm/src

# Install Dependencies
pip install vllm[audio] 
pip install flashinfer-python -i https://flashinfer.ai/whl/cu124/torch2.5

# set ENV for Faster model download from Huggingface.
export HF_TRANSFER = 1

# Launch the backend server
vllm serve Qwen/Qwen2-Audio-7B-Instruct \
    --trust-remote-code \
    --max-model-len 4096 \
    --limit-mm-per-prompt audio=1 \
    --gpu-memory-utilization 0.9 \
    --max-num-batched-tokens 4096
```

## Client Usage

```bash
# Process a request using a built-in audio asset
python main.py --audio-asset winning_call --prompt "Describe what you hear in this audio"

# Process a request using a local audio file
python main.py --audio-file your_audio.wav --prompt "Transcribe this audio"


# Process a request using Url.
python main.py --audio-url https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/guess_age_gender.wav --prompt "Describe this audio" 


# Run a high-concurrency benchmark to simulate production load
python main.py --audio-asset winning_call --simulate --concurrency 100 --requests 200 --output benchmark.txt

# Note: 
When running benchmarks with high concurrency settings, all output texts from the individual requests will be automatically saved. For example, if you specify --output benchmark.json, the benchmark metrics will be saved to that file, and all the individual text outputs will be saved in a directory called benchmark_results/ with files named output_1.txt, output_2.txt, etc.
```


```bash
# Example Output
============================================================
The speaker is female, based on the voice characteristics.
============================================================
Time: 4.62s, Generated 9 tokens
Speed: 1.95 tokens/sec
```

```bash
# Example Simulation Output:
============================================================
Simulation RESULTS
============================================================
Total requests:       200
Successful requests:  200
Failed requests:      0
Success rate:         100.00%
Duration:             38.96s
Request throughput:   5.13 req/s
Token throughput:     32.32 tokens/s
Avg tokens/request:   6.29
Avg request time:     19.03s
Avg tokens/second:    0.41
Min tokens/second:    0.04
Max tokens/second:    6.23
P50 tokens/second:    0.34
P90 tokens/second:    0.53
P99 tokens/second:    2.30


```



## Benchmark Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--concurrency` | Maximum number of concurrent requests | 10 |
| `--requests` | Total number of requests to send | 100 |
| `--rate` | Request rate in requests per second (0 = as fast as possible) | 0 |
| `--output` | File to save benchmark results (JSON format) | None |

## Example Benchmark Command

```bash
python main.py --audio-asset winning_call --prompt "Describe this audio" --simulate --concurrency 100 --requests 200
```
## credits
1. Claude-3.7 for sumamrization, generating valuable calaculation for metrics, for generating basic code snippets to support in success of the project. 
