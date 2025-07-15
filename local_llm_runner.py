"""
Local LLM Runner with Streaming Output and Performance Evaluation
Using Hugging Face Transformers
"""

import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import psutil
import threading
from datetime import datetime

# ===============================
# CONFIGURATION VARIABLES
# ===============================
MODEL_NAME = "google/gemma-3-1b-it"
MESSAGES = [
    {
        "role": "user",
        "content": "Simulate a conversation between a customer and a customer support executive at RAM Bank. The customer name is Darshil and the supports name is Sharon.\nDarshil is facing difficulty with logging into his netbanking. Sharon helps him successfully trouble shoot his steps."
    }
]
MAX_NEW_TOKENS = 2048  # Maximum number of new tokens to generate
TEMPERATURE = 0.7
DO_SAMPLE = True
TOP_P = 0.9
TOP_K = 50
DEVICE = "mps"  # "auto", "cpu", "cuda", or "mps"

# ===============================
# PERFORMANCE MONITORING
# ===============================
class PerformanceMonitor:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.memory_usage = []
        self.cpu_usage = []
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        self.start_time = time.time()
        self.monitoring = True
        self.memory_usage = []
        self.cpu_usage = []
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        self.end_time = time.time()
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
    
    def _monitor_resources(self):
        while self.monitoring:
            self.memory_usage.append(psutil.virtual_memory().percent)
            self.cpu_usage.append(psutil.cpu_percent(interval=0.1))
            time.sleep(0.1)
    
    def get_stats(self):
        total_time = self.end_time - self.start_time if self.end_time else 0
        avg_memory = sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0
        max_memory = max(self.memory_usage) if self.memory_usage else 0
        avg_cpu = sum(self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else 0
        max_cpu = max(self.cpu_usage) if self.cpu_usage else 0
        
        return {
            'total_time': total_time,
            'avg_memory_usage': avg_memory,
            'max_memory_usage': max_memory,
            'avg_cpu_usage': avg_cpu,
            'max_cpu_usage': max_cpu
        }

def detect_device():
    """Auto-detect the best available device"""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def print_separator(title=""):
    """Print a nice separator with optional title"""
    if title:
        print(f"\n{'='*20} {title} {'='*20}")
    else:
        print("="*60)

def print_config():
    """Print the current configuration"""
    print_separator("CONFIGURATION")
    print(f"Model: {MODEL_NAME}")
    print(f"Messages: {len(MESSAGES)} message(s)")
    for i, msg in enumerate(MESSAGES):
        print(f"  Message {i+1} - Role: {msg['role']}")
        print(f"  Content: {msg['content'][:100]}{'...' if len(msg['content']) > 100 else ''}")
    print(f"Max New Tokens: {MAX_NEW_TOKENS}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Top-p: {TOP_P}")
    print(f"Top-k: {TOP_K}")
    print(f"Device: {DEVICE}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def load_model_and_tokenizer():
    """Load the model and tokenizer"""
    print_separator("MODEL LOADING")
    
    # Determine device
    if DEVICE == "auto":
        device = detect_device()
    else:
        device = DEVICE
    
    print(f"Using device: {device}")
    
    load_start = time.time()
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device if device != "cpu" else None,
        trust_remote_code=True
    )
    
    if device == "cpu":
        model = model.to(device)
    
    load_time = time.time() - load_start
    
    print(f"Model loaded successfully in {load_time:.2f} seconds")
    print(f"Model device: {next(model.parameters()).device}")
    
    # Model info
    param_count = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {param_count:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model, tokenizer, device, load_time

def format_prompt(messages, tokenizer):
    """Format the messages using the tokenizer's chat template"""
    # Use the tokenizer's built-in chat template for proper formatting
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return formatted_prompt

def run_inference(model, tokenizer, device):
    """Run inference with streaming output"""
    print_separator("INFERENCE")
    
    # Format the prompt using chat template
    formatted_query = format_prompt(MESSAGES, tokenizer)
    print(f"Formatted prompt:\n{formatted_query}")
    print_separator("STREAMING OUTPUT")
    
    # Create text streamer for live output
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # Tokenize input
    inputs = tokenizer.encode(formatted_query, return_tensors="pt").to(device)
    
    # Start performance monitoring
    monitor = PerformanceMonitor()
    monitor.start_monitoring()
    
    inference_start = time.time()
    
    # Generate with streaming
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=DO_SAMPLE,
            top_p=TOP_P,
            top_k=TOP_K,
            streamer=streamer,
            pad_token_id=tokenizer.eos_token_id
        )
    
    inference_time = time.time() - inference_start
    monitor.stop_monitoring()
    
    # Get generated text (without prompt)
    generated_tokens = outputs[0][inputs.shape[1]:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return generated_text, inference_time, monitor, len(generated_tokens)

def print_evaluation(load_time, inference_time, monitor, num_generated_tokens):
    """Print detailed evaluation metrics"""
    print_separator("PERFORMANCE EVALUATION")
    
    stats = monitor.get_stats()
    tokens_per_second = num_generated_tokens / inference_time if inference_time > 0 else 0
    
    print(f"📊 TIMING METRICS:")
    print(f"  Model Loading Time: {load_time:.2f} seconds")
    print(f"  Inference Time: {inference_time:.2f} seconds")
    print(f"  Total Time: {load_time + inference_time:.2f} seconds")
    
    print(f"\n🚀 GENERATION METRICS:")
    print(f"  Generated Tokens: {num_generated_tokens}")
    print(f"  Tokens per Second: {tokens_per_second:.2f}")
    print(f"  Average Token Generation Time: {(inference_time/num_generated_tokens)*1000:.2f} ms")
    
    print(f"\n💻 RESOURCE USAGE:")
    print(f"  Average Memory Usage: {stats['avg_memory_usage']:.1f}%")
    print(f"  Peak Memory Usage: {stats['max_memory_usage']:.1f}%")
    print(f"  Average CPU Usage: {stats['avg_cpu_usage']:.1f}%")
    print(f"  Peak CPU Usage: {stats['max_cpu_usage']:.1f}%")
    
    print(f"\n🔧 SYSTEM INFO:")
    print(f"  Python Torch Version: {torch.__version__}")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA Device: {torch.cuda.get_device_name()}")
        print(f"  CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  MPS Available: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}")

def main():
    """Main execution function"""
    print_separator("LOCAL LLM RUNNER")
    print("🤖 Running Local LLM with Streaming Output and Performance Evaluation")
    
    try:
        # Print configuration
        print_config()
        
        # Load model and tokenizer
        model, tokenizer, device, load_time = load_model_and_tokenizer()
        
        # Run inference
        generated_text, inference_time, monitor, num_tokens = run_inference(model, tokenizer, device)
        
        # Print evaluation
        print_evaluation(load_time, inference_time, monitor, num_tokens)
        
        print_separator("COMPLETED SUCCESSFULLY")
        print("✅ LLM inference completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Inference interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
