"""Phase 3 Test 3.2: Test Auto-Initialization via env vars."""
from dotenv import load_dotenv
load_dotenv()

import os

print("=== Phase 3 Test 3.2: Environment Variables ===\n")

# Check env vars
print(f"OMIUM_API_KEY: {'***' + os.getenv('OMIUM_API_KEY', 'NOT SET')[-8:] if os.getenv('OMIUM_API_KEY') else 'NOT SET'}")
print(f"OMIUM_PROJECT: {os.getenv('OMIUM_PROJECT', 'NOT SET')}")
print(f"OMIUM_TRACING: {os.getenv('OMIUM_TRACING', 'NOT SET')}")
print(f"OMIUM_CHECKPOINTS: {os.getenv('OMIUM_CHECKPOINTS', 'NOT SET')}")

print("\n--- Initializing Omium ---")
import omium

try:
    config = omium.init()
    print(f"\n✓ Omium initialized successfully!")
    print(f"  Initialized: {omium.is_initialized()}")
    print(f"  Project: {config.project}")
    print(f"  Auto-trace: {config.auto_trace}")
    print(f"  Auto-checkpoint: {config.auto_checkpoint}")
    print(f"  Detected frameworks: {config.detected_frameworks}")
except Exception as e:
    print(f"\n✗ Initialization failed: {e}")
