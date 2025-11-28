import string
from collections import Counter
from typing import Dict, List, Tuple, Union, Callable

import json
import nltk
import numpy as np
import math
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


glue_qqp_dir = "data/QQP"
glove_path = "data/glove_6B/glove.6B.50d.txt"

all_tokens_path = "data/all_tokens.npy"
out_pairs_path = "data/out_pairs.npy"
emb_matrix_path = "data/emb_matrix.npy"
vocab_path = "data/vocab.json"
unk_words_path = "data/unk_words.npy"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        # import code

        # code.interact(local=locals())

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


class RankingDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        index_pairs_or_triplets: List[List[Union[str, float]]],
        idx_to_text_mapping: Dict[str, str],
        vocab: Dict[str, int],
        oov_val: int,
        preproc_func: Callable,
        max_len: int = 30,
    ):
        self.index_pairs_or_triplets = index_pairs_or_triplets
        self.idx_to_text_mapping = idx_to_text_mapping
        self.vocab = vocab
        self.oov_val = oov_val
        self.preproc_func = preproc_func
        self.max_len = max_len

    def __len__(self):
        return len(self.index_pairs_or_triplets)

    def _tokenized_text_to_index(self, tokenized_text: List[str]) -> List[int]:
        return [self.vocab.get(word, self.oov_val) for word in tokenized_text]

    def _convert_text_idx_to_token_idxs(self, idx: int) -> List[int]:
        text = self.idx_to_text_mapping[str(idx)]
        text = self.preproc_func(text)
        token_idxs = self._tokenized_text_to_index(text)
        return token_idxs

    def __getitem__(self, idx: int):
        pass


class TrainTripletsDataset(RankingDataset):
    def __getitem__(self, idx):
        val_pair = self.index_pairs_or_triplets[idx]
        q_idx = val_pair[0]
        d1_idx = val_pair[1]
        d2_idx = val_pair[2]
        label = val_pair[3]

        query = self._convert_text_idx_to_token_idxs(q_idx)[: self.max_len]
        document1 = self._convert_text_idx_to_token_idxs(d1_idx)[: self.max_len]
        document2 = self._convert_text_idx_to_token_idxs(d2_idx)[: self.max_len]

        return (
            {"query": query, "document": document1},
            {"query": query, "document": document2},
            label,
        )


class ValPairsDataset(RankingDataset):
    def __getitem__(self, idx):
        val_pair = self.index_pairs_or_triplets[idx]
        q_idx = val_pair[0]
        d_idx = val_pair[1]
        label = val_pair[2]

        query = self._convert_text_idx_to_token_idxs(q_idx)[: self.max_len]
        document = self._convert_text_idx_to_token_idxs(d_idx)[: self.max_len]
        return {"query": query, "document": document}, label


def collate_fn(batch_objs: List[Union[Dict[str, torch.Tensor], torch.FloatTensor]]):
    max_len_q1 = -1
    max_len_d1 = -1
    max_len_q2 = -1
    max_len_d2 = -1

    is_triplets = False
    for elem in batch_objs:
        if len(elem) == 3:
            left_elem, right_elem, label = elem
            is_triplets = True
        else:
            left_elem, label = elem

        max_len_q1 = max(len(left_elem["query"]), max_len_q1)
        max_len_d1 = max(len(left_elem["document"]), max_len_d1)
        if len(elem) == 3:
            max_len_q2 = max(len(right_elem["query"]), max_len_q2)
            max_len_d2 = max(len(right_elem["document"]), max_len_d2)

    q1s = []
    d1s = []
    q2s = []
    d2s = []
    labels = []

    for elem in batch_objs:
        if is_triplets:
            left_elem, right_elem, label = elem
        else:
            left_elem, label = elem

        pad_len1 = max_len_q1 - len(left_elem["query"])
        pad_len2 = max_len_d1 - len(left_elem["document"])
        if is_triplets:
            pad_len3 = max_len_q2 - len(right_elem["query"])
            pad_len4 = max_len_d2 - len(right_elem["document"])

        q1s.append(left_elem["query"] + [0] * pad_len1)
        d1s.append(left_elem["document"] + [0] * pad_len2)
        if is_triplets:
            q2s.append(right_elem["query"] + [0] * pad_len3)
            d2s.append(right_elem["document"] + [0] * pad_len4)
        labels.append([label])
    q1s = torch.LongTensor(q1s)
    d1s = torch.LongTensor(d1s)
    if is_triplets:
        q2s = torch.LongTensor(q2s)
        d2s = torch.LongTensor(d2s)
    labels = torch.FloatTensor(labels)

    ret_left = {"query": q1s, "document": d1s}
    if is_triplets:
        ret_right = {"query": q2s, "document": d2s}
        return ret_left, ret_right, labels
    else:
        return ret_left, labels


class Solution:
    def __init__(
        self,
        glue_qqp_dir: str,
        glove_vectors_path: str,
        min_token_occurancies: int = 1,
        random_seed: int = 0,
        emb_rand_uni_bound: float = 0.2,
        freeze_knrm_embeddings: bool = True,
        knrm_kernel_num: int = 21,
        knrm_out_mlp: List[int] = [10, 5],
        dataloader_bs: int = 1024,
        train_lr: float = 0.001,
        change_train_loader_ep: int = 10,
    ):
        self.glue_qqp_dir = glue_qqp_dir
        self.glove_vectors_path = glove_vectors_path
        self.glue_train_df = self.get_glue_df("train")
        self.glue_dev_df = self.get_glue_df("dev")
        self.dev_pairs_for_ndcg = self.create_val_pairs(self.glue_dev_df)
        self.min_token_occurancies = min_token_occurancies
        self.all_tokens = self.get_all_tokens(
            [self.glue_train_df, self.glue_dev_df], self.min_token_occurancies
        )

        self.random_seed = random_seed
        self.emb_rand_uni_bound = emb_rand_uni_bound
        self.freeze_knrm_embeddings = freeze_knrm_embeddings
        self.knrm_kernel_num = knrm_kernel_num
        self.knrm_out_mlp = knrm_out_mlp
        self.dataloader_bs = dataloader_bs
        self.train_lr = train_lr
        self.change_train_loader_ep = change_train_loader_ep

        self.model, self.vocab, self.unk_words = self.build_knrm_model()
        self.idx_to_text_mapping_train = self.get_idx_to_text_mapping(self.glue_train_df)
        self.idx_to_text_mapping_dev = self.get_idx_to_text_mapping(self.glue_dev_df)

        self.val_dataset = ValPairsDataset(
            self.dev_pairs_for_ndcg,
            self.idx_to_text_mapping_dev,
            vocab=self.vocab,
            oov_val=self.vocab["OOV"],
            preproc_func=self.simple_preproc,
        )
        self.val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.dataloader_bs,
            num_workers=0,
            collate_fn=collate_fn,
            shuffle=False,
        )

    def get_glue_df(self, partition_type: str) -> pd.DataFrame:
        print(f"Load {partition_type} dataset...")
        assert partition_type in ["dev", "train"]
        glue_df = pd.read_csv(
            self.glue_qqp_dir + f"/{partition_type}.tsv",
            sep="\t",
            on_bad_lines="skip",
            dtype=object,
        )
        glue_df = glue_df.dropna(axis=0, how="any").reset_index(drop=True)
        glue_df_fin = pd.DataFrame(
            {
                "id_left": glue_df["qid1"],
                "id_right": glue_df["qid2"],
                "text_left": glue_df["question1"],
                "text_right": glue_df["question2"],
                "label": glue_df["is_duplicate"].astype(int),
            }
        )
        return glue_df_fin

    def handle_punctuation(self, inp_str: str) -> str:
        for symbol in string.punctuation:
            inp_str = inp_str.replace(symbol, " ")
        return inp_str

    def simple_preproc(self, inp_str: str) -> List[str]:
        inp_str = self.handle_punctuation(inp_str)
        inp_str = inp_str.lower()
        return nltk.word_tokenize(inp_str)

    def _filter_rare_words(self, vocab: Dict[str, int], min_occurancies: int) -> List[str]:
        common_words = []
        for key in vocab:
            if vocab[key] >= min_occurancies:
                common_words.append(key)
        return common_words

    def get_all_tokens(self, list_of_df: List[pd.DataFrame], min_occurancies: int) -> List[str]:
        print("Get list of all tokens...")
        try:
            all_tokens = np.load(all_tokens_path, allow_pickle=True)
        except FileNotFoundError:
            pass
        else:
            all_tokens = list(all_tokens)
            return all_tokens

        all_tokens = []

        def flatten(lst):
            return [item for sublist in lst for item in sublist]

        for df in list_of_df:
            unique_text = set(df[["text_left", "text_right"]].values.reshape(-1))
            unique_text = flatten(map(self.simple_preproc, unique_text))
            all_tokens.extend(unique_text)

        all_tokens = self._filter_rare_words(Counter(all_tokens), min_occurancies)

        np.save(all_tokens_path, np.array(all_tokens, dtype=object))

        return all_tokens

    def _read_glove_embeddings(self, file_path: str) -> Dict[str, List[str]]:
        embeddings = {}
        with open(file_path, "r", encoding="utf8") as f:
            for line in f:
                parts = line.rstrip().split(" ")
                word = parts[0]
                vector = [float(x) for x in parts[1:]]
                embeddings[word] = vector
        return embeddings

    def create_glove_emb_from_file(
        self, file_path: str, inner_keys: List[str], random_seed: int, rand_uni_bound: float
    ) -> Tuple[np.ndarray, Dict[str, int], List[str]]:
        print("Create embeddings matrix...")
        try:
            emb_matrix = np.load(emb_matrix_path)
            vocab = json.load(
                open(vocab_path, "r", encoding="utf-8"),
            )
            unk_words = np.load(unk_words_path, allow_pickle=True)
        except FileNotFoundError:
            pass
        else:
            unk_words = list(unk_words)
            return emb_matrix, vocab, unk_words

        np.random.seed(random_seed)
        embeddings = self._read_glove_embeddings(file_path)
        d = len(list(embeddings.values())[0])
        vocab = {"PAD": 0, "OOV": 1}
        emb_matrix = [
            np.random.uniform(-rand_uni_bound, rand_uni_bound, d),
            np.random.uniform(-rand_uni_bound, rand_uni_bound, d),
        ]
        unk_words = ["PAD", "OOV"]
        for idx, word in enumerate(inner_keys, start=2):
            vocab[word] = idx
            if word in embeddings:
                emb_matrix.append(np.array(embeddings[word]))
            else:
                emb_matrix.append(np.random.uniform(-rand_uni_bound, rand_uni_bound, d))
                unk_words.append(word)
        emb_matrix = np.array(emb_matrix)
        emb_matrix[0] = np.zeros_like(emb_matrix[0])

        np.save(emb_matrix_path, emb_matrix)
        json.dump(
            vocab,
            open("data/vocab.json", "w", encoding="utf-8"),
            ensure_ascii=False,
            indent=4,
        )
        np.save(unk_words_path, np.array(unk_words))

        return emb_matrix, vocab, unk_words

    def build_knrm_model(self) -> Tuple[torch.nn.Module, Dict[str, int], List[str]]:
        emb_matrix, vocab, unk_words = self.create_glove_emb_from_file(
            self.glove_vectors_path, self.all_tokens, self.random_seed, self.emb_rand_uni_bound
        )
        torch.manual_seed(self.random_seed)
        knrm = KNRM(
            emb_matrix,
            freeze_embeddings=self.freeze_knrm_embeddings,
            out_layers=self.knrm_out_mlp,
            kernel_num=self.knrm_kernel_num,
        )
        return knrm, vocab, unk_words

    def sample_data_for_train_iter(
        self, inp_df: pd.DataFrame, seed: int
    ) -> List[List[Union[str, float]]]:
        print("Create sample data for train...")
        np.random.seed(seed)

        inp_df_select = inp_df[["id_left", "id_right", "label"]]

        ones = inp_df_select[inp_df_select["label"] == 1]
        zeroes = inp_df_select[inp_df_select["label"] == 0]

        merge_ll = pd.merge(ones, zeroes, on="id_left", suffixes=("_1", "_2"))
        merge_lr = pd.merge(
            ones, zeroes, left_on="id_left", right_on="id_right", suffixes=("_1", "_2")
        )
        merge_rr = pd.merge(ones, zeroes, on="id_right", suffixes=("_1", "_2"))
        merge_rl = pd.merge(
            ones, zeroes, left_on="id_right", right_on="id_left", suffixes=("_1", "_2")
        )

        merge_ll = merge_ll.rename(
            {"id_left": "query", "id_right_1": "doc_1", "id_right_2": "doc_2", "label_1": "label"},
            axis=1,
        )
        merge_lr = merge_lr.rename(
            {"id_left_1": "query", "id_right_1": "doc_1", "id_left_2": "doc_2", "label_1": "label"},
            axis=1,
        )
        merge_rr = merge_rr.rename(
            {"id_right": "query", "id_left_1": "doc_1", "id_left_2": "doc_2", "label_1": "label"},
            axis=1,
        )
        merge_rl = merge_rl.rename(
            {
                "id_right_1": "query",
                "id_left_1": "doc_1",
                "id_right_2": "doc_2",
                "label_1": "label",
            },
            axis=1,
        )
        result_dataset = pd.concat([merge_ll, merge_lr, merge_rr, merge_rl]).reset_index(drop=True)
        result_dataset = result_dataset[["query", "doc_1", "doc_2", "label"]].sample(10000)
        out_pairs = result_dataset.values.tolist()

        # add random negatives to each query
        groups = result_dataset.groupby("query")
        all_ids = (
            set(result_dataset["query"])
            .union(set(result_dataset["doc_1"]))
            .union(set(result_dataset["doc_2"]))
        )
        for query_id, _ in tqdm(groups):
            tmp_df = result_dataset[result_dataset["query"] == query_id]
            all_tmp_ids = set(tmp_df["doc_1"]).union(set(tmp_df["doc_2"])).union({query_id})
            all_negatives = list(all_ids - all_tmp_ids)
            for _ in range(len(tmp_df)):
                random_negative_id_1, random_negative_id_2 = np.random.choice(all_negatives, size=2)
                out_pairs.append([query_id, random_negative_id_1, random_negative_id_2, 0.5])
        return out_pairs

    def create_val_pairs(
        self, inp_df: pd.DataFrame, fill_top_to: int = 15, min_group_size: int = 2, seed: int = 0
    ) -> List[List[Union[str, float]]]:
        print("Create pairs for validation...")
        try:
            out_pairs = np.load(out_pairs_path, allow_pickle=True)
        except FileNotFoundError:
            pass
        else:
            out_pairs = list(out_pairs)
            return out_pairs

        inp_df_select = inp_df[["id_left", "id_right", "label"]]
        inf_df_group_sizes = inp_df_select.groupby("id_left").size()
        glue_dev_leftids_to_use = list(
            inf_df_group_sizes[inf_df_group_sizes >= min_group_size].index
        )
        groups = inp_df_select[inp_df_select.id_left.isin(glue_dev_leftids_to_use)].groupby(
            "id_left"
        )

        all_ids = set(inp_df["id_left"]).union(set(inp_df["id_right"]))

        out_pairs = []

        np.random.seed(seed)

        for id_left, group in groups:
            ones_ids = group[group.label > 0].id_right.values
            zeroes_ids = group[group.label == 0].id_right.values
            sum_len = len(ones_ids) + len(zeroes_ids)
            num_pad_items = max(0, fill_top_to - sum_len)
            if num_pad_items > 0:
                cur_chosen = set(ones_ids).union(set(zeroes_ids)).union({id_left})
                pad_sample = np.random.choice(
                    list(all_ids - cur_chosen), num_pad_items, replace=False
                ).tolist()
            else:
                pad_sample = []
            for i in ones_ids:
                out_pairs.append([id_left, i, 2])
            for i in zeroes_ids:
                out_pairs.append([id_left, i, 1])
            for i in pad_sample:
                out_pairs.append([id_left, i, 0])

        np.save(out_pairs_path, np.array(out_pairs, dtype=object))

        return out_pairs

    def get_idx_to_text_mapping(self, inp_df: pd.DataFrame) -> Dict[str, str]:
        left_dict = (
            inp_df[["id_left", "text_left"]]
            .drop_duplicates()
            .set_index("id_left")["text_left"]
            .to_dict()
        )
        right_dict = (
            inp_df[["id_right", "text_right"]]
            .drop_duplicates()
            .set_index("id_right")["text_right"]
            .to_dict()
        )
        left_dict.update(right_dict)
        return left_dict

    def ndcg_k(self, ys_true: np.array, ys_pred: np.array, ndcg_top_k: int = 10) -> float:
        def dcg(ys_true, ys_pred):
            argsort = np.argsort(ys_pred)[::-1]
            argsort = argsort[:ndcg_top_k]
            ys_true_sorted = ys_true[argsort]
            ret = 0
            for i, j in enumerate(ys_true_sorted, 1):
                ret += (2**j - 1) / math.log2(1 + i)
            return ret

        ideal_dcg = dcg(ys_true, ys_true)
        pred_dcg = dcg(ys_true, ys_pred)
        return pred_dcg / ideal_dcg

    def valid(self, model: torch.nn.Module, val_dataloader: torch.utils.data.DataLoader) -> float:
        labels_and_groups = val_dataloader.dataset.index_pairs_or_triplets
        labels_and_groups = pd.DataFrame(labels_and_groups, columns=["left_id", "right_id", "rel"])

        all_preds = []
        for batch in val_dataloader:
            inp_1, y = batch
            preds = model.predict(inp_1)
            preds_np = preds.cpu().detach().numpy()
            all_preds.append(preds_np)
        all_preds = np.concatenate(all_preds, axis=0)
        labels_and_groups["preds"] = all_preds

        ndcgs = []
        for cur_id in labels_and_groups.left_id.unique():
            cur_df = labels_and_groups[labels_and_groups.left_id == cur_id]
            ndcg = self.ndcg_k(cur_df.rel.values.reshape(-1), cur_df.preds.values.reshape(-1))
            if np.isnan(ndcg):
                ndcgs.append(0)
            else:
                ndcgs.append(ndcg)
        return np.mean(ndcgs)

    def train(self, n_epochs: int):
        print("Start training...")
        self.model.to(device)

        opt = torch.optim.SGD(self.model.parameters(), lr=self.train_lr)
        criterion = torch.nn.BCELoss()

        for ep in range(n_epochs):
            train_loss = 0.0

            if ep % self.change_train_loader_ep == 0:
                train_pairs = self.sample_data_for_train_iter(self.glue_train_df, seed=ep)

                train_dataset = TrainTripletsDataset(
                    train_pairs,
                    self.idx_to_text_mapping_train,
                    vocab=self.vocab,
                    oov_val=self.vocab["OOV"],
                    preproc_func=self.simple_preproc,
                )

                train_dataloader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=self.dataloader_bs,
                    num_workers=0,
                    collate_fn=collate_fn,
                    shuffle=True,
                )

            for batch in train_dataloader:
                inputs1, inputs2, labels = batch
                labels = labels.to(device)

                # opt.zero_grad()

                outputs = self.model(inputs1, inputs2)

                loss = criterion(outputs, labels)

                loss.backward()

                opt.step()

                train_loss += loss.item()

            # Print statistics per epoch
            avg_train_loss = train_loss / len(train_dataloader)

            # nDCG metric
            val_ndcg = self.valid(self.model, self.val_dataloader)

            print(
                f"Epoch [{ep + 1}/{n_epochs}], Train Loss: {avg_train_loss:.4f},Val nDCG@10: {val_ndcg:.4f}"
            )

            if val_ndcg > 0.94:
                break


if __name__ == "__main__":
    sol = Solution(glue_qqp_dir, glove_path, knrm_out_mlp=[])
    sol.train(20)

    state_mlp = sol.model.mlp.state_dict()
    torch.save(state_mlp, open("data/knrm_mlp.bin", "wb"))

    state_emb = sol.model.embeddings.state_dict()
    torch.save(state_emb, open("data/knrm_emb.bin", "wb"))
