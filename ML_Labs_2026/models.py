"""Small inspectable mathematical components for the guided experiments."""
import math
import numpy as np


def half_mse_gradient(x, y, w, b):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    residual = w * x + b - y
    loss = np.mean(residual ** 2) / 2
    return float(loss), float(np.mean(residual * x)), float(np.mean(residual))


def causal_attention(q, k, v):
    q, k, v = [np.asarray(a, dtype=float) for a in (q, k, v)]
    if q.ndim != 2 or q.shape != k.shape or v.ndim != 2 or len(v) != len(q):
        raise ValueError('Q/K: same (T,d); V: (T,dv)')
    scores = q @ k.T / math.sqrt(q.shape[1])
    mask = np.triu(np.ones(scores.shape, dtype=bool), k=1)
    scores[mask] = -np.inf
    weights = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    weights /= weights.sum(axis=1, keepdims=True)
    return weights @ v, weights


class TinyMLP:
    """Full-batch binary MLP: affine -> tanh -> affine -> sigmoid.

    The loss is mean binary cross-entropy; all updates use the same old weights.
    """
    def __init__(self, n_features=2, width=8, seed=7):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 1 / np.sqrt(n_features), (n_features, width))
        self.b1 = np.zeros(width)
        self.W2 = rng.normal(0, 1 / np.sqrt(width), width)
        self.b2 = np.array(0.)

    def forward(self, X):
        h = np.tanh(np.asarray(X) @ self.W1 + self.b1)
        z = h @ self.W2 + self.b2
        # exp(-logaddexp(0,-z)) computes the sigmoid stably.
        probability = np.exp(-np.logaddexp(0, -z))
        return h, z, probability

    def predict_proba(self, X):
        return self.forward(X)[2]

    def predict(self, X):
        return (self.predict_proba(X) >= .5).astype(int)

    def loss(self, X, y):
        _, z, _ = self.forward(X)
        return float(np.mean(np.logaddexp(0, z) - np.asarray(y) * z))

    def gradients(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        h, z, p = self.forward(X)
        dz = (p - y) / len(y)
        dW2, db2 = h.T @ dz, np.sum(dz)
        dh = dz[:, None] * self.W2[None, :]
        da = dh * (1 - h ** 2)
        return {'W1': X.T @ da, 'b1': da.sum(axis=0), 'W2': dW2, 'b2': db2}

    def step(self, X, y, rate):
        grads = self.gradients(X, y)
        for name, gradient in grads.items():
            setattr(self, name, getattr(self, name) - rate * gradient)

    @property
    def parameters(self):
        return self.W1.size + self.b1.size + self.W2.size + self.b2.size


class Bigram:
    """Empirical next-token counts; temperature rescales positive probabilities."""
    def __init__(self, sentences):
        self.counts = {}
        for sentence in sentences:
            words = ['<START>'] + sentence.split() + ['<END>']
            for current, following in zip(words, words[1:]):
                row = self.counts.setdefault(current, {})
                row[following] = row.get(following, 0) + 1

    def distribution(self, current, temperature=1.):
        if not 0 < temperature <= 5:
            raise ValueError('temperature in (0,5]')
        row = self.counts.get(current, {'<END>': 1})
        tokens = sorted(row)
        logits = np.log(np.array([row[token] for token in tokens], dtype=float)) / temperature
        p = np.exp(logits - max(logits)); p /= p.sum()
        return tokens, p

    def generate(self, start, temperature=1., max_tokens=12, seed=7):
        rng = np.random.default_rng(seed)
        words = [start]
        for _ in range(max_tokens):
            tokens, p = self.distribution(words[-1], temperature)
            next_token = str(rng.choice(tokens, p=p))
            if next_token == '<END>':
                break
            words.append(next_token)
        return ' '.join(words)
