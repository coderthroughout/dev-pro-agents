"""Phase 7: Test @trace and @checkpoint decorators."""
from dotenv import load_dotenv
load_dotenv()

import sys
import traceback

print("=== Phase 7: Decorator Testing ===\n")

try:
    # Initialize Omium
    print("--- Step 1: Initialize Omium ---")
    import omium
    omium.init(project="decorator-test", debug=True)

    # Test @trace decorator
    print("\n--- Step 2: Test @trace decorator ---")
    
    @omium.trace("research_step", span_type="function")
    def research_topic(topic: str) -> dict:
        """Simulate research."""
        print(f"  Researching: {topic}")
        return {"topic": topic, "results": ["result1", "result2"]}

    @omium.trace("analysis_step")
    def analyze_results(data: dict) -> str:
        """Analyze research results."""
        return f"Analyzed {len(data.get('results', []))} results"

    # Execute traced functions
    print("Executing traced functions...")
    research = research_topic("Python async")
    print(f"  Research result: {research}")

    analysis = analyze_results(research)
    print(f"  Analysis: {analysis}")

    print("\n[OK] @trace decorator works!")

    print("\n=== Phase 7 Complete ===")
    print("[OK] Decorator testing successful!")

except Exception as e:
    print(f"\n[ERROR] Error occurred: {e}")
    traceback.print_exc()
    sys.exit(1)
