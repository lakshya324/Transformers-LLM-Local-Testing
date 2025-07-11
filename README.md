# Local LLM Runner with Streaming Output

A Python script that runs local Large Language Models (LLMs) using Hugging Face Transformers with real-time streaming output and comprehensive performance evaluation.

## Features

- 🤖 **Local LLM Execution**: Run LLMs locally using Hugging Face Transformers
- 📡 **Live Streaming**: Real-time token-by-token output streaming to console
- 📊 **Performance Monitoring**: Detailed timing, resource usage, and generation metrics
- 🔧 **Easy Configuration**: All parameters configurable at the top of the script
- 🚀 **Device Auto-Detection**: Automatically detects and uses the best available device (CUDA/MPS/CPU)
- 💾 **Resource Monitoring**: Real-time CPU and memory usage tracking

## Quick Start

### 1. Setup Environment

```bash
# Run the setup script
./setup.sh

# Or manually:
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the LLM

```bash
# Activate virtual environment
source .venv/bin/activate

# Run the script
python local_llm_runner.py
```

## Configuration

Edit the variables at the top of `local_llm_runner.py` to customize your experience:

```python
# Model Configuration
MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"  # Change to any HF model
QUERY = "Explain the concept of machine learning in simple terms."  # Your question

# Generation Parameters
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.7
DO_SAMPLE = True
TOP_P = 0.9
TOP_K = 50
DEVICE = "auto"  # "auto", "cpu", "cuda", or "mps"
```

## Supported Models

The script works with any Hugging Face Causal Language Model. Some tested examples:

- `HuggingFaceTB/SmolLM2-135M-Instruct` (Default - Fast, lightweight)
- `HuggingFaceTB/SmolLM2-360M-Instruct`
- `HuggingFaceTB/SmolLM2-1.7B-Instruct`
- `microsoft/DialoGPT-medium`
- `gpt2`
- Any compatible model from Hugging Face Hub

## Output Example

```text
==================== CONFIGURATION ====================
Model: HuggingFaceTB/SmolLM2-135M-Instruct
Query: Explain the concept of machine learning in simple terms.
Max New Tokens: 200
Temperature: 0.7
Device: auto
Timestamp: 2025-07-09 15:30:45

==================== MODEL LOADING ====================
Using device: mps
Loading tokenizer...
Loading model...
Model loaded successfully in 2.34 seconds
Total parameters: 135,209,216

==================== STREAMING OUTPUT ====================
Machine learning is a branch of artificial intelligence that enables computers to learn and improve from data without being explicitly programmed for every task...

==================== PERFORMANCE EVALUATION ====================
📊 TIMING METRICS:
  Model Loading Time: 2.34 seconds
  Inference Time: 3.45 seconds
  Total Time: 5.79 seconds

🚀 GENERATION METRICS:
  Generated Tokens: 87
  Tokens per Second: 25.22
  Average Token Generation Time: 39.66 ms

💻 RESOURCE USAGE:
  Average Memory Usage: 45.2%
  Peak Memory Usage: 52.1%
  Average CPU Usage: 78.5%
  Peak CPU Usage: 95.2%
```

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Required packages (see `requirements.txt`)

## Device Support

- **CUDA**: Automatic GPU acceleration for NVIDIA GPUs
- **MPS**: Apple Silicon GPU acceleration (M1/M2 Macs)
- **CPU**: Fallback for systems without GPU acceleration

## Performance Tips

1. **Use GPU**: Enable CUDA or MPS for faster inference
2. **Smaller Models**: Start with smaller models like SmolLM2-135M for faster testing
3. **Adjust Tokens**: Reduce `MAX_NEW_TOKENS` for faster responses
4. **Temperature**: Lower temperature (0.1-0.3) for more deterministic outputs

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Try a smaller model or reduce `MAX_NEW_TOKENS`
2. **Slow Loading**: First-time model download can be slow; subsequent runs are faster
3. **Import Errors**: Ensure all requirements are installed: `pip install -r requirements.txt`

### Model Download

Models are automatically downloaded from Hugging Face Hub on first use and cached locally.

## License

This project is licensed under the MIT License - see the LICENSE file for details.