"""Phase 3 Test 3.3: Test LangGraph Instrumentation Detection."""
from dotenv import load_dotenv
load_dotenv()

print("=== Phase 3 Test 3.3: LangGraph Instrumentation ===\n")

# Initialize Omium first
import omium
config = omium.init(debug=True)

print(f"\n--- Config ---")
print(f"Project: {config.project}")
print(f"Detected frameworks: {config.detected_frameworks}")

# Check if LangGraph is instrumented
print(f"\n--- Instrumentation Check ---")
try:
    from omium.integrations.langgraph import is_instrumented
    print(f"LangGraph instrumented: {is_instrumented()}")
except Exception as e:
    print(f"Error checking instrumentation: {e}")

print("\n=== Test 3.3 Complete ===")
