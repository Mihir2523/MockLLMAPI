"""
MockLLM Demo — Test it live!

Steps:
  1. Run this script: python demo.py
  2. A browser window will open to Gemini
  3. Make sure you're logged in (if not, log in now)
  4. Press Enter in this terminal when ready
  5. Watch the magic happen!
"""
import sys
sys.path.insert(0, ".")

from mocklm import MockLLM

def main():
    print("=" * 55)
    print("  MockLLM Demo - Browser-Based Free LLM API")
    print("=" * 55)

    # Step 1: Create the client
    print("\n[1] Creating MockLLM client with Gemini provider...")
    llm = MockLLM(provider="gemini", headless=False)

    # Step 2: Start the browser
    print("[2] Launching browser...\n")
    llm.start()

    print("-" * 55)
    print("  Browser is open!")
    print("  Make sure you are LOGGED IN to Gemini.")
    print("  If not, log in now in the browser window.")
    print("-" * 55)
    input("\n>>> Press ENTER when you are logged in and ready... ")

    # Step 3: Send a test query
    print("\n[3] Sending test query: 'What is Python in 2 sentences?'")
    print("    Waiting for response...\n")

    response = llm.chat("What is Python in 2 sentences?")

    # Step 4: Show the response
    print("=" * 55)
    print("  RESPONSE RECEIVED!")
    print("=" * 55)

    print(f"\n--- OpenAI-Compatible Format ---")
    print(f"  response.id:      {response.id}")
    print(f"  response.model:   {response.model}")
    print(f"  response.object:  {response.object}")
    print(f"  finish_reason:    {response.choices[0].finish_reason}")
    print(f"  role:             {response.choices[0].message.role}")
    print(f"  success:          {response.success}")
    print(f"  elapsed:          {response.elapsed:.1f}s")

    print(f"\n--- Response Text ---")
    print(f"  {response.choices[0].message.content}")

    print(f"\n--- Shortcut ---")
    print(f"  response.text = {response.text}")

    # Step 5: Try another query?
    print("\n" + "-" * 55)
    again = input(">>> Send another query? (type it or press ENTER to quit): ")
    if again.strip():
        print("    Waiting for response...\n")
        resp2 = llm.chat(again.strip())
        print(f"  Response: {resp2.text}")

    # Cleanup
    print("\n[4] Closing browser...")
    llm.close()
    print("\nDone! MockLLM works.")

if __name__ == "__main__":
    main()
