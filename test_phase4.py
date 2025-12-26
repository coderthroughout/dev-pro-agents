"""Phase 4: Test Explicit omium.init() with Supervisor."""
from dotenv import load_dotenv
load_dotenv()

import sys
import traceback

print("=== Phase 4: Explicit Init with Supervisor ===\n")

try:
    # Initialize Omium BEFORE importing Supervisor
    print("--- Step 1: Initializing Omium ---")
    import omium

    config = omium.init(
        project="explicit-init-test",
        debug=True
    )

    print(f"\n[OK] Omium initialized")
    print(f"  Project: {config.project}")
    print(f"  Auto-trace: {config.auto_trace}")
    print(f"  Detected frameworks: {config.detected_frameworks}")

    # Now import Supervisor (after init)
    print("\n--- Step 2: Creating Supervisor ---")
    from src.supervisor import Supervisor

    supervisor = Supervisor()
    print(f"\n[OK] Supervisor created successfully")
    print(f"  App compiled: {supervisor.app is not None}")
    print(f"  Research agent: {supervisor.research_agent is not None}")
    print(f"  Coding agent: {supervisor.coding_agent is not None}")

    # Verify instrumentation is still active
    print("\n--- Step 3: Checking Instrumentation ---")
    from omium.integrations.langgraph import is_instrumented
    print(f"LangGraph instrumented: {is_instrumented()}")

    print("\n=== Phase 4 Complete ===")
    print("[OK] Explicit init with Supervisor works!")

except Exception as e:
    print(f"\n[ERROR] Error occurred: {e}")
    traceback.print_exc()
    sys.exit(1)
