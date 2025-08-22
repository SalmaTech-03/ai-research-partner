# test_search.py

import os
from dotenv import load_dotenv

# --- THIS IS THE CORRECT, MODERN IMPORT ---
from langchain_community.utilities import GoogleSearchAPIWrapper

print("--- Testing Google Search API Wrapper ---")

try:
    secrets_path = os.path.join(os.path.dirname(__file__), '.streamlit', 'secrets.toml')
    load_dotenv(dotenv_path=secrets_path)
    print("Successfully loaded credentials from .streamlit/secrets.toml")
except Exception as e:
    print(f"Could not load secrets file: {e}")
    exit()

if not os.getenv("GOOGLE_CSE_ID") or not os.getenv("GOOGLE_API_KEY"):
    print("\nERROR: GOOGLE_CSE_ID or GOOGLE_API_KEY not found.")
    exit()

try:
    print("\nInitializing GoogleSearchAPIWrapper...")
    search = GoogleSearchAPIWrapper()
    
    query = "latest AI research papers on knowledge graphs"
    print(f"Executing search for: '{query}'")
    
    results = search.run(query)
    
    print("\n--- Search Results ---")
    print(results)
    print("\n--- Test Successful ---")

except Exception as e:
    print(f"\n--- An Error Occurred ---")
    print(f"Error Details: {e}")