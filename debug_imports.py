
import traceback
import sys

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

print("1. Importing huggingface_hub...")
try:
    import huggingface_hub
    print(f"   Success! Version: {huggingface_hub.__version__}")
    print(f"   is_offline_mode available: {hasattr(huggingface_hub, 'is_offline_mode')}")
except Exception:
    traceback.print_exc()

print("2. Importing transformers...")
try:
    import transformers
    print(f"   Success! Version: {transformers.__version__}")
except Exception:
    traceback.print_exc()

print("3. Importing diffusers...")
try:
    import diffusers
    print(f"   Success! Version: {diffusers.__version__}")
except Exception:
    traceback.print_exc()
