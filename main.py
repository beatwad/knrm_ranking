from typing import Dict, List

import os
import json
import torch
from flask import Flask, request, jsonify
from lingua import Language, LanguageDetectorBuilder
import string
import threading
import dotenv
import faiss
import numpy as np
import torch.nn.functional as F


dotenv.load_dotenv()

app = Flask(__name__)

helper = None

ann_k = 100
ret_k = 10

languages = [
    Language.ENGLISH,
    Language.RUSSIAN,
    Language.SPANISH,
    Language.FRENCH,
    Language.GERMAN,
]
lang_detector = LanguageDetectorBuilder.from_languages(*languages).build()


class GaussianKernel(torch.nn.Module):
    def __init__(self, mu: float = 1.0, sigma: float = 1.0):
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def forward(self, x):
        return torch.exp(-0.5 * ((x - self.mu) ** 2) / self.sigma**2)


class KNRM(torch.nn.Module):
    def __init__(
        self,
        emb_state_dict,
        freeze_embeddings: bool,
        kernel_num: int = 21,
        sigma: float = 0.1,
        exact_sigma: float = 0.001,
        out_layers: List[int] = [10, 5],
    ):
        super().__init__()

        vocab_size, dim = emb_state_dict["weight"].shape
        self.embeddings = torch.nn.Embedding(vocab_size, dim)
        self.embeddings.load_state_dict(emb_state_dict)

        self.kernel_num = kernel_num
        self.sigma = sigma
        self.exact_sigma = exact_sigma
        self.out_layers = out_layers

        self.kernels = self._get_kernels_layers()

        self.mlp = self._get_mlp()

        self.out_activation = torch.nn.Sigmoid()

    def _get_kernels_layers(self) -> torch.nn.ModuleList:
        kernels = []

        for i in range(self.kernel_num - 1):
            mu = 1 / (self.kernel_num - 1) + 2 * i / (self.kernel_num - 1) - 1
            kernels.append((mu, self.sigma))
        kernels.append((1, self.exact_sigma))

        kernels = torch.nn.ModuleList([GaussianKernel(k[0], k[1]) for k in kernels])
        return kernels

    def _get_mlp(self) -> torch.nn.Sequential:
        layers = []
        layer_dims = [self.kernel_num] + self.out_layers + [1]

        for i in range(len(layer_dims) - 1):
            layers.append(torch.nn.Linear(layer_dims[i], layer_dims[i + 1]))
            layers.append(torch.nn.ReLU())

        return torch.nn.Sequential(*layers[:-1])

    def forward(
        self, input_1: Dict[str, torch.Tensor], input_2: Dict[str, torch.Tensor]
    ) -> torch.FloatTensor:
        logits_1 = self.predict(input_1)
        logits_2 = self.predict(input_2)

        logits_diff = logits_1 - logits_2

        out = self.out_activation(logits_diff)

        return out

    def _get_matching_matrix(self, query: torch.Tensor, doc: torch.Tensor) -> torch.FloatTensor:
        # query: [Batch, Left]
        # doc: [Batch, Right]
        q = self.embeddings.weight[query]
        d = self.embeddings.weight[doc]
        # q: [Batch, Left, Dim]
        # d: [Batch, Right, Dim]
        q = q.unsqueeze(2)
        d = d.unsqueeze(1)
        # output [Batch, Left, Right]
        return F.cosine_similarity(q, d, dim=-1)

    def _apply_kernels(self, matching_matrix: torch.FloatTensor) -> torch.FloatTensor:
        KM = []
        for kernel in self.kernels:
            # shape = [B]
            K = torch.log1p(kernel(matching_matrix).sum(dim=-1)).sum(dim=-1)
            KM.append(K)
        # shape = [B, K]
        kernels_out = torch.stack(KM, dim=1)
        return kernels_out

    def predict(self, inputs: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        # shape = [Batch, Left], [Batch, Right]
        query, doc = inputs["query"], inputs["document"]
        # shape = [Batch, Left, Right]
        matching_matrix = self._get_matching_matrix(query, doc)
        # shape = [Batch, Kernels]
        kernels_out = self._apply_kernels(matching_matrix)
        # shape = [Batch]
        out = self.mlp(kernels_out)
        return out


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

    def update_index(self, documents: Dict[str, str]):
        glove_embeddings = self.glove_embeddings
        document_embeddings = []
        knrm_index = []
        # Prepare FAISS and KNRM ind
        for document in documents.values():
            processed_document = self._simple_preproc(document).replace("  ", " ")
            indexed_document = [
                self.vocab.get(word, self.vocab["OOV"]) for word in processed_document.split(" ")
            ]
            vector = self._transform_to_glove_embedding(glove_embeddings, processed_document)
            if vector is not None:
                document_embeddings.append(vector)
                knrm_index.append(indexed_document)
        # Create FAISS index
        dim = len(list(glove_embeddings.values())[0])
        index = faiss.IndexFlatL2(dim)
        index.add(np.vstack(document_embeddings))
        # Update the helper
        self.faiss_index = index
        self.documents_ids = list(documents.keys())
        self.documents = list(documents.values())
        self.knrm_index = knrm_index
        return index.ntotal

    def get_suggestions(self, queries: List[str], ann_k: int = 50):
        lang_check = []
        suggestions = []
        glove_embeddings = self.glove_embeddings
        for query in queries:
            # Language check
            is_en = self._language_check(query)
            if not is_en:
                suggestions.append(None)
                lang_check.append(False)
                continue
            lang_check.append(True)
            # Preprocess the query
            processed_query = self._simple_preproc(query)
            vector = self._transform_to_glove_embedding(glove_embeddings, processed_query)
            # Search k nearest neighbors in FAISS index
            _, ann = self.faiss_index.search(vector.reshape(1, -1), ann_k)
            ann = ann.ravel()
            # Select corresponding document embeddings from KNRM index and remove duplicates
            selected_documents = list(set(tuple(self.knrm_index[ann_idx]) for ann_idx in ann))
            # Pad the documents to the same length
            emb_lens = [len(selected_document) for selected_document in selected_documents]
            max_len = max(emb_lens)
            selected_documents = [
                list(selected_document) + [self.vocab["PAD"]] * (max_len - len(selected_document))
                for selected_document in selected_documents
            ]
            # Transform the query and documents to tensors
            embedded_query = torch.LongTensor(
                [self.vocab.get(word, self.vocab["OOV"]) for word in processed_query]
            ).unsqueeze(0)
            embedded_query = embedded_query.repeat(len(selected_documents), 1)
            selected_documents = torch.LongTensor(selected_documents)
            # Rank the documents using KNRM model
            scores = []
            inputs = dict()
            inputs["query"] = embedded_query
            inputs["document"] = selected_documents
            scores = self.model.predict(inputs)
            scores = scores.detach().numpy().ravel()
            score_idxs = np.argsort(scores)[::-1]
            suggestion = []
            # Get the suggestions
            for ann_idx in ann[score_idxs[:ret_k]]:
                suggestion.append((self.documents_ids[ann_idx], self.documents[ann_idx]))
            suggestions.append(suggestion)
        return suggestions, lang_check

    def _handle_punctuation(self, inp_str: str) -> str:
        for symbol in string.punctuation:
            inp_str = inp_str.replace(symbol, " ")
        return inp_str

    def _simple_preproc(self, inp_str: str) -> List[str]:
        inp_str = self._handle_punctuation(inp_str)
        inp_str = inp_str.lower()
        return inp_str

    def _transform_to_glove_embedding(
        self, glove_embeddings: Dict[str, List[int]], inp_str: str
    ) -> np.array:
        processed_document = self._handle_punctuation(inp_str).lower()
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

    def _language_check(self, query: str) -> bool:
        language = lang_detector.detect_language_of(query)
        try:
            language = language.iso_code_639_1.name
            is_en = language == "EN"
        except Exception:
            is_en = False
        return is_en


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
    suggestions, lang_check = helper.get_suggestions(queries, ann_k)
    return jsonify(suggestions=suggestions, lang_check=lang_check)


@app.route("/update_index", methods=["POST"])
def update_index():
    data = request.get_json()
    documents = data["documents"]
    index_size = helper.update_index(documents)
    return jsonify(status="ok", index_size=index_size)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=11000)
