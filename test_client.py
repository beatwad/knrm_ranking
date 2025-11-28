import requests
import time
import sys
import json

BASE_URL = "http://0.0.0.0:11000"

def test_ping():
    print(f"Testing /ping endpoint at {BASE_URL}/ping...")
    try:
        response = requests.get(f"{BASE_URL}/ping")
        if response.status_code == 200:
            data = response.json()
            print(f"Ping response: {data}")
            return data.get("status")
        else:
            print(f"Ping returned status code: {response.status_code}")
            return None
    except requests.exceptions.ConnectionError:
        print("Server is not reachable.")
        return None

def test_update_index():
    print("\nTesting /update_index endpoint...")
    # Create some dummy documents
    documents = {
        "doc1": "The quick brown fox jumps over the lazy dog",
        "doc2": "Python is a powerful programming language used for web development and data science",
        "doc3": "Deep learning is a subset of machine learning based on artificial neural networks",
        "doc4": "Flask is a micro web framework written in Python",
        "doc5": "Natural language processing enables computers to understand human language"
    }
    
    payload = {"documents": documents}
    try:
        response = requests.post(f"{BASE_URL}/update_index", json=payload)
        response.raise_for_status()
        data = response.json()
        print(f"Update index response: {data}")
        
        if data.get("status") == "ok" and data.get("index_size") == len(documents):
            print("Index update successful.")
            return True
        else:
            print("Index update failed or size mismatch.")
            return False
    except Exception as e:
        print(f"Update index failed: {e}")
        return False

def test_query():
    print("\nTesting /query endpoint...")
    # Queries: valid English, and one Spanish to test language detection
    queries = [
        "programming in python", 
        "neural networks and deep learning", 
        "quick brown fox", 
        "Hola mundo como estas"
    ]

    payload = {"queries": queries}
    try:
        response = requests.post(f"{BASE_URL}/query", json=payload)
        response.raise_for_status()
        data = response.json()
        
        suggestions = data.get("suggestions")
        lang_check = data.get("lang_check")
        
        print(f"Language check results: {lang_check}")
        
        if len(suggestions) != len(queries):
            print(f"Error: Expected {len(queries)} suggestions, got {len(suggestions)}")
            return False
            
        all_passed = True
        for i, q in enumerate(queries):
            print(f"\nQuery: '{q}'")
            is_en = lang_check[i]
            print(f"Detected as English: {is_en}")
            
            if is_en:
                res = suggestions[i]
                if not isinstance(res, list):
                     print("Error: Suggestions should be a list of tuples.")
                     all_passed = False
                     continue
                print(f"Top suggestions (count {len(res)}):")
                for doc_id, doc_text in res:
                    print(f" - {doc_id}: {doc_text}")
            else:
                print("No suggestions expected (language check failed).")
                if suggestions[i] is not None:
                    print("Error: Suggestions should be None for failed lang check.")
                    all_passed = False

        return all_passed
    except Exception as e:
        print(f"Query failed: {e}")
        return False

def main():
    print("--- Starting Test Client ---")
    print("Waiting for server to be ready...")
    
    # Wait loop for server initialization
    max_retries = 20
    for i in range(max_retries):
        status = test_ping()
        if status == "ok":
            print("Server is ready and initialized!")
            break
        elif status == "in_progress":
            print("Server is initializing (helper loading)... waiting 2s")
            time.sleep(2)
        else:
            print("Server not reachable yet... waiting 2s")
            time.sleep(2)
    else:
        print("Timeout: Server failed to become ready or is not running.")
        print("Please make sure you run 'python solution.py' in a separate terminal.")
        sys.exit(1)

    if not test_update_index():
        print("Aborting tests due to update_index failure.")
        sys.exit(1)
        
    if not test_query():
        print("Tests failed during query execution.")
        sys.exit(1)
        
    print("\n--- All tests passed successfully! ---")

if __name__ == "__main__":
    main()

