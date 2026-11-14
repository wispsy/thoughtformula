import math
import random

random.seed(42)

# =====================================================================
# 1. AUTOGRAD ENGINE (Scalar Automatic Differentiation)
# =====================================================================
class Value:
    """Stores a single scalar value and its gradient in a computation graph."""
    __slots__ = ('data', 'grad', '_backward', '_prev', 'm', 'v')

    def __init__(self, data, _children=()):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = _children
        # Adam optimizer state
        self.m = 0.0
        self.v = 0.0

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        out = Value(self.data ** other, (self,))
        def _backward():
            self.grad += (other * (self.data ** (other - 1))) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,))
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(max(self.data, 1e-12)), (self,))
        def _backward():
            self.grad += (1.0 / max(self.data, 1e-12)) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,))
        def _backward():
            if self.data > 0:
                self.grad += out.grad
        out._backward = _backward
        return out

    def __neg__(self): return self * -1.0
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * (other ** -1.0)

    def backward(self):
        # Iterative topological sort to avoid Python recursion limits
        topo, visited = [], set()
        stack = [(self, False)]
        while stack:
            node, processed = stack.pop()
            if node in visited:
                continue
            if processed:
                visited.add(node)
                topo.append(node)
            else:
                stack.append((node, True))
                for child in node._prev:
                    if child not in visited:
                        stack.append((child, False))
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()


# =====================================================================
# 2. PURE PYTHON MATRIX & NEURAL NETWORK PRIMITIVES
# =====================================================================
def rand_matrix(rows, cols, std=0.08):
    return [[Value(random.gauss(0, std)) for _ in range(cols)] for _ in range(rows)]

def rand_vector(size):
    return [Value(0.0) for _ in range(size)]

def linear(x, weight, bias=None):
    """Multiplies a vector x by weight matrix W and adds optional bias."""
    out = []
    for i, w_row in enumerate(weight):
        start = bias[i] if bias else Value(0.0)
        out.append(sum((wi * xi for wi, xi in zip(w_row, x)), start))
    return out

def rmsnorm(x):
    """Root Mean Square Layer Normalization."""
    ms = sum((xi ** 2 for xi in x), Value(0.0)) * (1.0 / len(x))
    inv_rms = (ms + 1e-5) ** -0.5
    return [xi * inv_rms for xi in x]

def softmax(logits):
    """Numerically stable Softmax over a list of Value objects."""
    max_val = max(v.data for v in logits)
    exps = [(v - max_val).exp() for v in logits]
    total = sum(exps, Value(0.0))
    return [e / total for e in exps]


# =====================================================================
# 3. DECODER-ONLY TRANSFORMER (MICRO-GPT)
# =====================================================================
class MicroGPT:
    def __init__(self, vocab_size, n_embd=12, block_size=8):
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.block_size = block_size

        # Token & positional embedding tables
        self.tok_emb = rand_matrix(vocab_size, n_embd)
        self.pos_emb = rand_matrix(block_size, n_embd)

        # Causal Self-Attention weights (Q, K, V, Output projection)
        self.wq = rand_matrix(n_embd, n_embd)
        self.wk = rand_matrix(n_embd, n_embd)
        self.wv = rand_matrix(n_embd, n_embd)
        self.wo = rand_matrix(n_embd, n_embd)

        # Feed-Forward Network (MLP) weights
        self.w_fc1 = rand_matrix(2 * n_embd, n_embd)
        self.b_fc1 = rand_vector(2 * n_embd)
        self.w_fc2 = rand_matrix(n_embd, 2 * n_embd)
        self.b_fc2 = rand_vector(n_embd)

        # Language Model Head (projects embeddings to vocabulary logits)
        self.lm_head = rand_matrix(vocab_size, n_embd)
        self.b_head = rand_vector(vocab_size)

    def parameters(self):
        matrices = [
            self.tok_emb, self.pos_emb,
            self.wq, self.wk, self.wv, self.wo,
            self.w_fc1, self.w_fc2, self.lm_head
        ]
        vectors = [self.b_fc1, self.b_fc2, self.b_head]
        params = [p for mat in matrices for row in mat for p in row]
        params.extend([p for vec in vectors for p in vec])
        return params

    def forward(self, idx_seq):
        T = len(idx_seq)
        scale = 1.0 / math.sqrt(self.n_embd)

        # 1. Token + Positional Embeddings
        x = [
            [self.tok_emb[idx_seq[t]][i] + self.pos_emb[t][i] for i in range(self.n_embd)]
            for t in range(T)
        ]

        # 2. Causal Self-Attention Block
        x_norm = [rmsnorm(xt) for xt in x]
        Q = [linear(xt, self.wq) for xt in x_norm]
        K = [linear(xt, self.wk) for xt in x_norm]
        V = [linear(xt, self.wv) for xt in x_norm]

        for t in range(T):
            # Causal masking: token t only attends to tokens 0..t
            scores = [
                sum((Q[t][i] * K[k][i] for i in range(self.n_embd)), Value(0.0)) * scale
                for k in range(t + 1)
            ]
            probs = softmax(scores)
            attn_t = [
                sum((probs[k] * V[k][i] for k in range(t + 1)), Value(0.0))
                for i in range(self.n_embd)
            ]
            proj_t = linear(attn_t, self.wo)
            # Residual connection
            x[t] = [x[t][i] + proj_t[i] for i in range(self.n_embd)]

        # 3. Feed-Forward MLP Block
        for t in range(T):
            xt_norm = rmsnorm(x[t])
            hidden = [v.relu() for v in linear(xt_norm, self.w_fc1, self.b_fc1)]
            mlp_out = linear(hidden, self.w_fc2, self.b_fc2)
            # Residual connection
            x[t] = [x[t][i] + mlp_out[i] for i in range(self.n_embd)]

        # 4. Final Norm & Output Logits
        logits = [linear(rmsnorm(x[t]), self.lm_head, self.b_head) for t in range(T)]
        return logits

    def generate(self, start_tokens, max_new_tokens, temperature=0.5):
        idx = list(start_tokens)
        for _ in range(max_new_tokens):
            context = idx[-self.block_size:]
            logits = self.forward(context)[-1]
            # Extract raw floats for inference (no gradient tracking needed)
            scaled = [v.data / max(temperature, 1e-4) for v in logits]
            max_l = max(scaled)
            probs = [math.exp(s - max_l) for s in scaled]
            next_id = random.choices(range(self.vocab_size), weights=probs)[0]
            idx.append(next_id)
        return idx


# =====================================================================
# 4. DATASET, TRAINING LOOP & GENERATION
# =====================================================================
if __name__ == "__main__":
    # Tiny repetitive corpus so the micro-model learns patterns in seconds
    corpus = (
        "deep learning in pure python. "
        "small models learn patterns fast. "
        "attention is all you need. "
    ) * 4

    # Character-level tokenizer
    chars = sorted(list(set(corpus)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])

    data = encode(corpus)
    block_size = 8
    model = MicroGPT(vocab_size=vocab_size, n_embd=12, block_size=block_size)
    params = model.parameters()

    print(f"Vocab size: {vocab_size} characters | Total parameters: {len(params)}")
    print("Training Micro-GPT in pure Python...\n")

    steps = 250
    lr = 0.03

    for step in range(1, steps + 1):
        # Sample a random chunk of text
        ix = random.randint(0, len(data) - block_size - 1)
        x_seq = data[ix : ix + block_size]
        y_seq = data[ix + 1 : ix + block_size + 1]

        # Forward pass & Cross-Entropy Loss
        logits = model.forward(x_seq)
        loss = Value(0.0)
        for t in range(block_size):
            probs = softmax(logits[t])
            loss = loss - probs[y_seq[t]].log()
        loss = loss * (1.0 / block_size)

        # Backward pass
        loss.backward()

        # Adam Optimizer Step
        for p in params:
            p.m = 0.9 * p.m + 0.1 * p.grad
            p.v = 0.999 * p.v + 0.001 * (p.grad ** 2)
            m_hat = p.m / (1.0 - 0.9 ** step)
            v_hat = p.v / (1.0 - 0.999 ** step)
            p.data -= lr * m_hat / (math.sqrt(v_hat) + 1e-8)
            p.grad = 0.0

        if step % 50 == 0 or step == 1:
            print(f"Step {step:3d}/{steps} | Loss: {loss.data:.4f}")

    # Autoregressive Inference
    print("\n--- Generated Output ---")
    for prompt in ["deep ", "small ", "atten"]:
        out_tokens = model.generate(encode(prompt), max_new_tokens=30, temperature=0.3)
        print(f"Prompt '{prompt}' -> {decode(out_tokens)}")
