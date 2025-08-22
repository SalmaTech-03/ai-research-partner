# find_class.py

print("--- Running Diagnostic Script ---")

try:
    import langchain_arangodb
    print("\nSuccessfully imported 'langchain_arangodb'")
    print("Contents of 'langchain_arangodb':")
    print(dir(langchain_arangodb))
except ImportError as e:
    print(f"\nFAILED to import 'langchain_arangodb'. Error: {e}")

try:
    import langchain_arangodb.graphs
    print("\nSuccessfully imported 'langchain_arangodb.graphs'")
    print("Contents of 'langchain_arangodb.graphs':")
    print(dir(langchain_arangodb.graphs))
except (ImportError, AttributeError) as e:
    print(f"\nFAILED to import or access 'langchain_arangodb.graphs'. Error: {e}")

print("\n--- Diagnostic Complete ---")