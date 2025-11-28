# Neural Matching System

A document matching system using Kernel Neural Ranking Model (KNRM) and GloVe embeddings with FAISS index, served via a Flask REST API. Features language filtering.

## Project Structure

- `main.py`: Flask server with KNRM model and API endpoints.
- `train.py`: Script for training the KNRM model using the QQP dataset.
- `test_client.py`: End-to-end test script using QQP dataset.
- `data/`: Required directory for model artifacts and datasets.

## Requirements

- Python 3.7+
- `flask`, `torch`, `numpy`, `faiss-cpu`, `lingua-language-detector`, `python-dotenv`
- Test client: `pandas`, `requests`

## Setup

1. **Install dependencies:**
   ```bash
   pip install flask torch numpy faiss-cpu lingua-language-detector python-dotenv pandas requests
   ```

2. **Prepare Data:**
   Ensure the `data/` directory contains:
   - `glove_6B/glove.6B.50d.txt`
   - `vocab.json`
   - `knrm_emb.bin`
   - `knrm_mlp.bin`
   - `QQP/dev.tsv` (for testing)

   Download the data from [here](https://dl.fbaipublicfiles.com/glue/data/QQP-clean.zip) and place it in the `data/QQP/` directory.

3. **Environment Variables:**
   Create a `.env` file or export:
   ```env
   EMB_PATH_GLOVE=data/glove_6B/glove.6B.50d.txt
   VOCAB_PATH=data/vocab.json
   EMB_PATH_KNRM=data/knrm_emb.bin
   MLP_PATH=data/knrm_mlp.bin
   ```

## Usage

### Train Model
To retrain the KNRM model:
```bash
python train.py
```
This script loads QQP data and GloVe embeddings, trains the ranking model using triplet loss, and saves artifacts (`knrm_emb.bin`, `knrm_mlp.bin`, `vocab.json`) to `data/`.

GloVe embeddings can be downloaded from [here](http://nlp.stanford.edu/data/glove.6B.zip).

### Start Server
```bash
python main.py
```
Server runs at `http://0.0.0.0:11000`. Initialization may take a moment.

### Run Tests
```bash
python test_client.py
```
This script waits for the server, updates the index with QQP data, and runs validation queries.

## API Endpoints

- **`GET /ping`**: Health check. Returns `{"status": "ok"}` when ready.
- **`POST /update_index`**: Index documents.
  - Body: `{"documents": {"id": "text", ...}}`
- **`POST /query`**: Retrieve ranked suggestions.
  - Body: `{"queries": ["query text", ...]}`
  - Returns: List of top ranked `[doc_id, text]` and language check results.
