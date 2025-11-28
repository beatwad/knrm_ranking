from typing import Dict, List

import os
import json
import torch
from flask import Flask, request, jsonify
from langdetect import detect
from knrm import KNRM
import string
import threading
import dotenv
import faiss
import numpy as np


dotenv.load_dotenv()

app = Flask(__name__)

helper = None
k = 10


class Helper:
    def __init__(self):
        self.emb_path_glove = os.getenv("EMB_PATH_GLOVE")
        self.vocab_path = os.getenv("VOCAB_PATH")
        self.emb_path_knrm = os.getenv("EMB_PATH_KNRM")
        self.mlp_path = os.getenv("MLP_PATH")

        self.prepare_model()
        self.glove_embeddings = self._read_glove_embeddings(self.emb_path_glove)
        self.vocab = self._load_vocab(self.vocab_path)

        self.documents = None
        self.knrm_index = None
        self.faiss_index = None

    def prepare_model(self):
        self.model = KNRM(
            emb_state_dict=torch.load(self.emb_path_knrm),
            freeze_embeddings=True,
            out_layers=[],
            kernel_num=21,
            sigma=0.1,
            exact_sigma=0.001,
        )
        self.model.mlp.load_state_dict(torch.load(self.mlp_path))

        global model_is_ready
        model_is_ready = True

    def handle_punctuation(self, inp_str: str) -> str:
        for symbol in string.punctuation:
            inp_str = inp_str.replace(symbol, " ")
        return inp_str

    def simple_preproc(self, inp_str: str) -> List[str]:
        inp_str = self.handle_punctuation(inp_str)
        inp_str = inp_str.lower()
        return inp_str

    def transform_to_glove_embedding(
        self, glove_embeddings: Dict[str, List[int]], inp_str: str
    ) -> np.array:
        processed_document = self.handle_punctuation(inp_str).lower()
        vector = [
            glove_embeddings[tok]
            for tok in processed_document.replace("  ", " ").split(" ")
            if tok in glove_embeddings
        ]
        if vector:
            vector = np.mean(vector, axis=0)
            return vector
        return None

    def _read_glove_embeddings(self, file_path: str) -> Dict[str, List[str]]:
        glove_embeddings = {}
        with open(file_path, "r", encoding="utf8") as f:
            for line in f:
                parts = line.rstrip().split(" ")
                word = parts[0]
                vector = [float(x) for x in parts[1:]]
                glove_embeddings[word] = vector
        return glove_embeddings

    def _load_vocab(self, file_path: str) -> Dict[str, int]:
        with open(file_path, "r") as f_in:
            vocab = json.load(f_in)
            return vocab


def init_helper():
    global helper
    helper = Helper()


with app.app_context():
    thread = threading.Thread(target=init_helper)
    thread.daemon = True
    thread.start()


@app.route("/ping")
def ping():
    if helper:
        return jsonify(status="ok")
    return jsonify(status="in_progress")


@app.route("/query", methods=["POST"])
def query():
    if not helper.faiss_index:
        return jsonify(status="FAISS is not initialized!")
    data = request.get_json()
    queries = data["queries"]
    lang_check = []
    suggestions = []
    glove_embeddings = helper.glove_embeddings
    for query in queries:
        # Language check
        is_en = detect(query) == "en"
        lang_check.append(is_en)
        if not is_en:
            suggestions.append(None)
            continue
        # Preprocess the query
        processed_query = helper.simple_preproc(query)
        vector = helper.transform_to_glove_embedding(glove_embeddings, processed_query)
        # Search k nearest neighbors in FAISS index
        _, ann = helper.faiss_index.search(vector.reshape(1, -1), k)
        ann = ann.ravel()
        # Embed the query

        # Select corresponding document embeddings from KNRM index and remove duplicates
        selected_documents = set(tuple(helper.knrm_index[ann_idx]) for ann_idx in ann)
        selected_documents = [list(document) for document in selected_documents]
        # Pad the documents to the same length
        emb_lens = [len(selected_document) for selected_document in selected_documents]
        max_len = max(emb_lens)
        selected_documents = [
            selected_document + [helper.vocab["PAD"]] * (max_len - len(selected_document))
            for selected_document in selected_documents
        ]

        embedded_query = torch.LongTensor(
            [helper.vocab.get(word, helper.vocab["OOV"]) for word in processed_query]
        ).unsqueeze(0)
        embedded_query = embedded_query.repeat(len(selected_documents), 1)
        selected_documents = torch.LongTensor(selected_documents)

        import code

        code.interact(local={**globals(), **locals()})

        # Rank the documents using KNRM model
        scores = []
        inputs = dict()
        inputs["query"] = embedded_query
        inputs["document"] = selected_documents
        scores = helper.model.predict(inputs)
        scores = scores.detach().numpy().ravel()
        score_idxs = np.argsort(scores)[::-1]
        suggestion = []

        for ann_idx in ann[score_idxs]:
            suggestion.append((helper.documents_ids[ann_idx], helper.documents[ann_idx]))

        suggestions.append(suggestion)

    return jsonify(suggestions=suggestions, lang_check=lang_check)


@app.route("/update_index", methods=["POST"])
def update_index():
    data = request.get_json()
    documents = data["documents"]
    glove_embeddings = helper.glove_embeddings
    document_embeddings = []
    knrm_index = []
    # Prepare FAISS and KNRM ind
    for document in documents.values():
        processed_document = helper.simple_preproc(document).replace("  ", " ")
        indexed_document = [
            helper.vocab.get(word, helper.vocab["OOV"]) for word in processed_document.split(" ")
        ]

        vector = helper.transform_to_glove_embedding(glove_embeddings, processed_document)
        if vector is not None:
            document_embeddings.append(vector)
            knrm_index.append(indexed_document)

    dim = len(list(glove_embeddings.values())[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.vstack(document_embeddings))

    # Update the helper
    helper.faiss_index = index
    helper.documents_ids = list(documents.keys())
    helper.documents = list(documents.values())
    helper.knrm_index = knrm_index

    return jsonify(status="ok", index_size=index.ntotal)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=11000)
