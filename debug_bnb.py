
try:
    import bitsandbytes
    print(f"bitsandbytes version: {bitsandbytes.__version__}")
    print("Success! bitsandbytes is importable.")
except ImportError:
    print("bitsandbytes not installed.")
except Exception as e:
    print(f"bitsandbytes failed to import: {e}")
