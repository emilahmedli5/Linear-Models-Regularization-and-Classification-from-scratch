import numpy as np
import re
from collections import Counter

class BagOfWords:
    def __init__(self, max_features: int = 5000):
        self.max_features = max_features
        self.vocab = {}
        
    def _tokenize(self, text: str) -> list:
        # Simple tokenization: lowercase and extract alphabetic words
        return re.findall(r'\b[a-z]{2,}\b', text.lower())
        
    def fit(self, raw_documents: list) -> "BagOfWords":
        """Builds a vocabulary of the top words by frequency."""
        word_counts = Counter()
        for doc in raw_documents:
            word_counts.update(self._tokenize(doc))
            
        # Select the top 'max_features' words
        top_words = [word for word, count in word_counts.most_common(self.max_features)]
        
        # Create a mapping of word to index
        self.vocab = {word: i for i, word in enumerate(top_words)}
        return self
        
    def transform(self, raw_documents: list) -> np.ndarray:
        """Converts documents to count vectors."""
        X = np.zeros((len(raw_documents), len(self.vocab)))
        for i, doc in enumerate(raw_documents):
            tokens = self._tokenize(doc)
            for token in tokens:
                if token in self.vocab:
                    X[i, self.vocab[token]] += 1
        return X

class TfidfTransformer:
    def __init__(self):
        self.idf = None
        
    def fit(self, X: np.ndarray) -> "TfidfTransformer":
        """Calculates the IDF weights."""
        N = X.shape[0]
        # Count number of documents where term t appears (f_{t,d} > 0)
        df = np.sum(X > 0, axis=0)
        
        # Calculate IDF: log(N / (1 + df))
        self.idf = np.log(N / (1 + df))
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Converts count vectors to TF-IDF weights."""
        # tf-idf = f_{t,d} * idf_t
        return X * self.idf