import requests
import time
import sys
import pandas as pd
import numpy as np

BASE_URL = "http://0.0.0.0:11000"
DATA_PATH = "data/QQP/dev.tsv"


def load_data():
    print(f"Loading data from {DATA_PATH}...")
    try:
        df = pd.read_csv(DATA_PATH, sep="\t")
        # Filter is_duplicate = 1
        df = df[df["is_duplicate"] == 1]
        df = df.dropna(subset=["question1", "question2", "qid1"])

        # Create documents dict: qid1 -> question1
        # Convert qid1 to string for JSON compatibility
        documents = pd.Series(df.question1.values, index=df.qid1.astype(str)).to_dict()
        print(f"Loaded {len(documents)} unique documents from duplicate pairs.")

        return df, documents
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)


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


def test_update_index(documents):
    print("\nTesting /update_index endpoint...")

    payload = {"documents": documents}
    try:
        response = requests.post(f"{BASE_URL}/update_index", json=payload)
        response.raise_for_status()
        data = response.json()
        print(f"Update index response: {data}")

        if data.get("status") == "ok":
            # We don't strictly check size equality here because duplicates in source might have been merged
            print(f"Index update successful. Size: {data.get('index_size')}")
            return True
        else:
            print("Index update failed.")
            return False
    except Exception as e:
        print(f"Update index failed: {e}")
        return False


def test_query(df, N_queries):
    print("\nTesting /query endpoint...")

    # Select N_queries random rows from the filtered dataframe
    if len(df) < N_queries:
        print("Not enough data to sample N queries.")
        return False

    np.random.seed(42)
    sample = df.sample(N_queries)
    queries = sample["question2"].tolist()
    expected_ids = sample["qid1"].astype(str).tolist()

    payload = {"queries": queries}
    try:
        response = requests.post(f"{BASE_URL}/query", json=payload)
        response.raise_for_status()
        data = response.json()

        suggestions = data.get("suggestions")
        lang_check = data.get("lang_check")

        if len(suggestions) != len(queries):
            print(f"Error: Expected {len(queries)} suggestions, got {len(suggestions)}")
            return False

        all_passed = True
        found_n_queries = 0
        rejected_count = 0

        for i, (q, expected_id) in enumerate(zip(queries, expected_ids)):
            print(f"\nQuery: '{q}'")
            print(f"Expected ID: {expected_id}")

            # Check language if present
            if lang_check and not lang_check[i]:
                print("Warning: Query detected as not English (unexpected for QQP).")

            res = suggestions[i]
            if res is None:
                print("Error: Suggestions is None.")
                rejected_count += 1
                continue

            if not isinstance(res, list):
                print("Error: Suggestions should be a list of tuples.")
                all_passed = False
                continue

            # Found IDs in top suggestions
            found_ids = [str(r[0]) for r in res]
            print(f"Top suggestions IDs: {found_ids}")

            if expected_id in found_ids:
                print("SUCCESS: Expected ID found in suggestions.")
                found_n_queries += 1
            else:
                print("FAILURE: Expected ID NOT found in suggestions.")

        accuracy = found_n_queries / N_queries
        print(f"Found {found_n_queries} out of {N_queries} queries.")
        print(f"Accuracy: {accuracy * 100:.2f}%")

        if accuracy < 0.5:
            print("FAILURE: Accuracy is less than 50%.")
            all_passed = False

        reject_ratio = rejected_count / N_queries
        print(f"Rejected {rejected_count} out of {N_queries} queries.")
        print(f"Reject ratio: {reject_ratio * 100:.2f}%")
        if reject_ratio > 0.05:
            print("FAILURE: Reject ratio is greater than 5%.")
            all_passed = False

        return all_passed
    except Exception as e:
        print(f"Query failed: {e}")
        return False


def test_non_english_queries():
    print("\nTesting /query endpoint with non-English queries...")
    non_english_queries = [
        "Привет, как дела?",
        "Hola, ¿cómo estás?",
        "Bonjour, comment ça va?",
        "Hallo, wie geht es dir?",
        "Ciao, come stai?",
        "Olá, como vai?",
        "你好吗",
        "お元気ですか",
        "안녕하세요",
        "مرحباً، كيف حالك؟",
    ]
    payload = {"queries": non_english_queries}
    try:
        response = requests.post(f"{BASE_URL}/query", json=payload)
        response.raise_for_status()
        data = response.json()
        lang_check = data.get("lang_check")
        if lang_check is None:
            print("Error: lang_check field missing in response.")
            return False
        rejected_count = sum(1 for check in lang_check if not check)
        print(f"Rejected {rejected_count} out of {len(non_english_queries)} non-English queries.")
        if rejected_count >= 9:
            print("SUCCESS: Non-English filtering works as expected.")
            return True
        else:
            print("FAILURE: Too many non-English queries were accepted.")
            return False
    except Exception as e:
        print(f"Non-English query test failed: {e}")
        return False


print("--- Starting Test Client ---")

# Load data first
df, documents = load_data()

print("Waiting for server to be ready...")

# Wait loop for server initialization
max_retries = 60
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

if not test_update_index(documents):
    print("Aborting tests due to update_index failure.")
    sys.exit(1)

if not test_query(df, N_queries=100):
    print("Tests failed during query execution.")
    sys.exit(1)

if not test_non_english_queries():
    print("Tests failed during non-English query execution.")
    sys.exit(1)

print("\n--- All tests passed successfully! ---")
