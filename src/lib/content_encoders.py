"""Content encoders behind a single interface.

Every encoder exposes:
    fit(documents)        -> fit on training documents only (leak-free); no-op for
                             pre-trained models
    transform(documents)  -> L2-normalised matrix (n_documents, dim)

Available names: 'tfidf', 'minilm', 'bge-small', 'bge-base', 'gte-small', 'gte-base',
'gist', 'mxbai', plus the fine-tuned encoders 'finetuned-minilm' / 'finetuned-bge-base'
(produced by src/finetune_encoder.py). Add new ones in ENCODERS.
"""

import os

import numpy as np
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

DIM_TFIDF = 128
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')


def device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


class TfidfEncoder:
    """TF-IDF followed by a truncated SVD (classic reference)."""

    name = 'tfidf'

    def __init__(self, dim=DIM_TFIDF, seed=0):
        self.vectorizer = TfidfVectorizer(min_df=5, sublinear_tf=True, lowercase=True,
                                          strip_accents='unicode', max_features=300000)
        self.svd = TruncatedSVD(n_components=dim, random_state=seed)
        self.fitted = False

    def fit(self, documents):
        matrix = self.vectorizer.fit_transform(documents)
        self.svd.fit(matrix)
        self.fitted = True
        return self

    def transform(self, documents):
        if not self.fitted:
            self.fit(documents)
        return normalize(self.svd.transform(self.vectorizer.transform(documents))).astype(np.float32)


class SentenceEncoder:
    """Any sentence-transformers model (BGE, GTE, MiniLM, mxbai, ...)."""

    def __init__(self, model_name, name=None, batch_size=64):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device())
        self.name = name or model_name.split('/')[-1]
        self.batch_size = batch_size

    def fit(self, documents):
        return self

    def transform(self, documents):
        if not documents:
            return np.zeros((0, self.model.get_sentence_embedding_dimension()), dtype=np.float32)
        vectors = self.model.encode(documents, batch_size=self.batch_size, show_progress_bar=False,
                                    convert_to_numpy=True, normalize_embeddings=True)
        return vectors.astype(np.float32)


SENTENCE_MODELS = {
    'minilm': 'sentence-transformers/all-MiniLM-L6-v2',
    'bge-small': 'BAAI/bge-small-en-v1.5',
    'bge-base': 'BAAI/bge-base-en-v1.5',
    'bge-large': 'BAAI/bge-large-en-v1.5',
    'gte-small': 'thenlper/gte-small',
    'gte-base': 'thenlper/gte-base',
    'gist': 'avsolatorio/GIST-small-Embedding-v0',
    'mxbai': 'mixedbread-ai/mxbai-embed-large-v1',
}


FINETUNED_MODELS = {
    'finetuned-minilm': os.path.join(MODELS_DIR, 'finetuned-minilm'),
    'finetuned-bge-base': os.path.join(MODELS_DIR, 'finetuned-bge-base'),
}


def encode_grouped(encoder, groups):
    """One vector per group of texts: every text is encoded on its own, then averaged.

    Empty groups become zero vectors. This is the alternative to concatenating a user's posts
    into one long string, which the encoder then truncates at its token limit.
    """
    flat = [text for group in groups for text in group]
    if not flat:
        return np.zeros((len(groups), 1), dtype=np.float32)
    encoded = encoder.transform(flat)
    pooled = np.zeros((len(groups), encoded.shape[1]), dtype=np.float32)
    position = 0
    for index, group in enumerate(groups):
        if group:
            pooled[index] = encoded[position:position + len(group)].mean(axis=0)
            position += len(group)
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    return pooled / np.maximum(norms, 1e-8)


def get_encoder(name):
    if name == 'tfidf':
        return TfidfEncoder()
    if name in SENTENCE_MODELS:
        return SentenceEncoder(SENTENCE_MODELS[name], name=name)
    if name in FINETUNED_MODELS:
        path = FINETUNED_MODELS[name]
        if not os.path.isdir(path):
            raise ValueError(f'{name} not found at {path}: run src/finetune_encoder.py first')
        return SentenceEncoder(path, name=name)
    raise ValueError(f'unknown encoder: {name}')


def available():
    return ['tfidf'] + list(SENTENCE_MODELS) + list(FINETUNED_MODELS)
