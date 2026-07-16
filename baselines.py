"""
Baseline scheme simulations for comparison with zkPMT.

Each baseline faithfully models the cost structure described in the paper:

  Garg    – zkPoT-style: independent R1CS circuit + Groth16 key generation
            per round. O(T·g) initialization, O(g) per-round prove/verify.

  Kaizen  – GKR-based recursive proof with Pedersen commitments and
            Merkle verification paths. O(T·log n) init, O(log n) prove/verify.

  zkCNN   – Forward-only inference verification. O(g) init, no training proof.

  zkDL    – Forward-propagation only (no backprop). O(g) init, O(g) prove,
            O(1) verify. Does NOT cover gradient updates.

  VeriCNN – End-to-end CNN training verification with per-round complex
            circuits (conv + BN layers). O(g) per round, O(T·g) cumulative.

All costs are simulated via the same calibrated timing constants as zkpmt.py,
scaled by the complexity factors described in the paper.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from fixed_point import poseidon_hash, encode_model_weights

# ── Shared timing constants (must match zkpmt.py) ────────────────────────────
GATE_COST_SETUP  = 1e-4
GATE_COST_PROVE  = 2e-4
GROTH16_VERIFY   = 0.012
POSEIDON_PER_PARAM = 1e-7

# Kaizen-specific: GKR polynomial commitment overhead
GKR_LOG_FACTOR   = 0.6    # relative to Groth16 per gate
PEDERSEN_COST    = 0.002  # per commitment round

# VeriCNN: extra overhead for conv/BN circuit modules
VERICNN_OVERHEAD = 1.15   # 15% more than Garg per round


def _count_gates(model: nn.Module) -> int:
    total_params = sum(p.numel() for p in model.parameters())
    relu_gates = 0
    conv_gates = 0

    for module in model.modules():
        if isinstance(module, nn.Linear):
            relu_gates += module.out_features * 2
        elif isinstance(module, nn.Conv2d):
            conv_gates += module.weight.numel() * 10
        elif isinstance(module, nn.BatchNorm2d):
            conv_gates += module.num_features * 32 * 32 * 2

    if any(isinstance(m, nn.Conv2d) for m in model.modules()):
        relu_gates = total_params // 2

    poseidon_gates = (total_params // 32 + 1) * 128
    return total_params + relu_gates + conv_gates + poseidon_gates


def _sgd_step(model, X_batch, y_batch, lr):
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()
    optimizer.zero_grad()
    loss = criterion(model(X_batch), y_batch)
    loss.backward()
    optimizer.step()


# ─────────────────────────────────────────────────────────────────────────────
class GargScheme:
    """
    zkPoT-style verifiable training (Garg et al., CCS 2023).
    Constructs a separate R1CS circuit and calls Groth16 key generation
    for EACH training round → O(T·g) initialization overhead.
    """

    def __init__(self, model, dataset, lr=0.01, batch_size=32, num_rounds=100):
        self.model = model
        self.X_train, self.y_train = dataset
        self.lr = lr
        self.batch_size = batch_size
        self.num_rounds = num_rounds
        self.num_gates = _count_gates(model)
        self.num_params = sum(p.numel() for p in model.parameters())

        self.init_time = 0.0
        self.prove_times = []
        self.verify_times = []

    def setup(self):
        """O(T·g): compile circuit + key generation for every round."""
        t0 = time.perf_counter()
        # Simulate per-round circuit compilation and key generation
        time.sleep(self.num_rounds * self.num_gates * GATE_COST_SETUP)
        self.init_time = time.perf_counter() - t0
        return self.init_time

    def run(self):
        n = len(self.X_train)
        for i in range(self.num_rounds):
            idx = torch.randint(0, n, (self.batch_size,))
            _sgd_step(self.model, self.X_train[idx], self.y_train[idx], self.lr)

            # Per-round: full Groth16 prove
            t0 = time.perf_counter()
            time.sleep(self.num_gates * GATE_COST_PROVE)
            self.prove_times.append(time.perf_counter() - t0)

            # Per-round: full Groth16 verify
            t0 = time.perf_counter()
            time.sleep(GROTH16_VERIFY)
            self.verify_times.append(time.perf_counter() - t0)

        # Storage: independent proof per round
        proof_bytes = self.num_rounds * 192
        circuit_bytes = self.num_rounds * self.num_params * 4
        storage_mb = (proof_bytes + circuit_bytes) / (1024 ** 2)

        return {
            "scheme": "Garg",
            "init_time_s": self.init_time,
            "single_prove_time_s": float(np.mean(self.prove_times)),
            "total_prove_time_s": float(np.sum(self.prove_times)),
            "verify_time_s": float(np.sum(self.verify_times)),
            "storage_mb": storage_mb,
        }


# ─────────────────────────────────────────────────────────────────────────────
class KaizenScheme:
    """
    Kaizen (Abbaszadeh et al., CCS 2024).
    GKR-based recursive proof with Pedersen commitments.
    O(T·log n) initialization; O(log n) per-round prove/verify.
    Faster per-round than Garg due to compressed subcircuits,
    but higher initialization due to Merkle path construction.
    """

    def __init__(self, model, dataset, lr=0.01, batch_size=32, num_rounds=100):
        self.model = model
        self.X_train, self.y_train = dataset
        self.lr = lr
        self.batch_size = batch_size
        self.num_rounds = num_rounds
        self.num_params = sum(p.numel() for p in model.parameters())
        self.log_n = max(1, int(np.log2(self.num_params)))

        self.init_time = 0.0
        self.prove_times = []
        self.verify_times = []

    def setup(self):
        """O(T·log n): Pedersen commitments + Merkle path per round."""
        t0 = time.perf_counter()
        time.sleep(self.num_rounds * self.log_n * PEDERSEN_COST * GKR_LOG_FACTOR)
        self.init_time = time.perf_counter() - t0
        return self.init_time

    def run(self):
        n = len(self.X_train)
        for i in range(self.num_rounds):
            idx = torch.randint(0, n, (self.batch_size,))
            _sgd_step(self.model, self.X_train[idx], self.y_train[idx], self.lr)

            # O(log n) prove
            t0 = time.perf_counter()
            time.sleep(self.log_n * GATE_COST_PROVE * GKR_LOG_FACTOR)
            self.prove_times.append(time.perf_counter() - t0)

            # O(log n) verify
            t0 = time.perf_counter()
            time.sleep(self.log_n * GROTH16_VERIFY * GKR_LOG_FACTOR * 0.1)
            self.verify_times.append(time.perf_counter() - t0)

        proof_bytes = self.num_rounds * 192
        storage_mb = proof_bytes / (1024 ** 2) * self.log_n

        return {
            "scheme": "Kaizen",
            "init_time_s": self.init_time,
            "single_prove_time_s": float(np.mean(self.prove_times)),
            "total_prove_time_s": float(np.sum(self.prove_times)),
            "verify_time_s": float(np.sum(self.verify_times)),
            "storage_mb": storage_mb,
        }


# ─────────────────────────────────────────────────────────────────────────────
class zkCNNScheme:
    """
    zkCNN (Liu et al., CCS 2021).
    Forward-inference verification only. Does NOT cover gradient updates
    or parameter continuity. Included as a reference for circuit efficiency.
    O(g) initialization; no training proof generation.
    """

    def __init__(self, model, dataset, lr=0.01, batch_size=32, num_rounds=100):
        self.model = model
        self.X_train, self.y_train = dataset
        self.lr = lr
        self.batch_size = batch_size
        self.num_rounds = num_rounds
        self.num_gates = _count_gates(model)
        self.num_params = sum(p.numel() for p in model.parameters())

        self.init_time = 0.0

    def setup(self):
        """O(g): compile forward-only circuit (no backprop constraints)."""
        t0 = time.perf_counter()
        # Forward circuit is simpler: ~60% of full circuit
        time.sleep(self.num_gates * GATE_COST_SETUP * 0.6)
        self.init_time = time.perf_counter() - t0
        return self.init_time

    def run(self):
        # zkCNN does not generate training proofs; only forward inference
        n = len(self.X_train)
        for i in range(self.num_rounds):
            idx = torch.randint(0, n, (self.batch_size,))
            _sgd_step(self.model, self.X_train[idx], self.y_train[idx], self.lr)
            # No proof generation

        storage_mb = (self.num_params * 4) / (1024 ** 2)  # only model weights

        return {
            "scheme": "zkCNN",
            "init_time_s": self.init_time,
            "single_prove_time_s": 0.0,   # not applicable
            "total_prove_time_s": 0.0,
            "verify_time_s": 0.0,
            "storage_mb": storage_mb,
        }


# ─────────────────────────────────────────────────────────────────────────────
class zkDLScheme:
    """
    zkDL (Sun et al., IEEE TIFS 2025).
    Optimized for forward-propagation verification using sumcheck and
    lookup techniques. Does NOT cover backpropagation or gradient updates.
    O(g) init; O(g) prove; O(1) verify.
    """

    def __init__(self, model, dataset, lr=0.01, batch_size=32, num_rounds=100):
        self.model = model
        self.X_train, self.y_train = dataset
        self.lr = lr
        self.batch_size = batch_size
        self.num_rounds = num_rounds
        self.num_gates = _count_gates(model)
        self.num_params = sum(p.numel() for p in model.parameters())

        self.init_time = 0.0
        self.prove_times = []
        self.verify_times = []

    def setup(self):
        """O(g): hard-coded forward circuit compilation."""
        t0 = time.perf_counter()
        time.sleep(self.num_gates * GATE_COST_SETUP * 0.65)
        self.init_time = time.perf_counter() - t0
        return self.init_time

    def run(self):
        n = len(self.X_train)
        for i in range(self.num_rounds):
            idx = torch.randint(0, n, (self.batch_size,))
            _sgd_step(self.model, self.X_train[idx], self.y_train[idx], self.lr)

            # Forward-only proof: ~80% of full Groth16 cost
            t0 = time.perf_counter()
            time.sleep(self.num_gates * GATE_COST_PROVE * 0.80)
            self.prove_times.append(time.perf_counter() - t0)

            # O(1) verify
            t0 = time.perf_counter()
            time.sleep(GROTH16_VERIFY)
            self.verify_times.append(time.perf_counter() - t0)

        storage_mb = (self.num_params * 4) / (1024 ** 2)

        return {
            "scheme": "zkDL",
            "init_time_s": self.init_time,
            "single_prove_time_s": float(np.mean(self.prove_times)),
            "total_prove_time_s": float(np.sum(self.prove_times)),
            "verify_time_s": float(np.sum(self.verify_times)),
            "storage_mb": storage_mb,
        }


# ─────────────────────────────────────────────────────────────────────────────
class VeriCNNScheme:
    """
    VeriCNN – end-to-end zero-knowledge verification for CNN training.
    Constructs a complete zk-SNARK proof per round including conv,
    batch-norm, and FC layers. O(g) per round; O(T·g) cumulative storage.
    """

    def __init__(self, model, dataset, lr=0.01, batch_size=32, num_rounds=100):
        self.model = model
        self.X_train, self.y_train = dataset
        self.lr = lr
        self.batch_size = batch_size
        self.num_rounds = num_rounds
        self.num_gates = _count_gates(model)
        self.num_params = sum(p.numel() for p in model.parameters())

        self.init_time = 0.0
        self.prove_times = []
        self.verify_times = []

    def setup(self):
        """O(g): layer-by-layer circuit with feature-map hashing."""
        t0 = time.perf_counter()
        time.sleep(self.num_gates * GATE_COST_SETUP * VERICNN_OVERHEAD)
        self.init_time = time.perf_counter() - t0
        return self.init_time

    def run(self):
        n = len(self.X_train)
        for i in range(self.num_rounds):
            idx = torch.randint(0, n, (self.batch_size,))
            _sgd_step(self.model, self.X_train[idx], self.y_train[idx], self.lr)

            # Full circuit per round with extra conv/BN overhead
            t0 = time.perf_counter()
            time.sleep(self.num_gates * GATE_COST_PROVE * VERICNN_OVERHEAD)
            self.prove_times.append(time.perf_counter() - t0)

            # O(1) verify (fixed-structure circuit)
            t0 = time.perf_counter()
            time.sleep(GROTH16_VERIFY * VERICNN_OVERHEAD)
            self.verify_times.append(time.perf_counter() - t0)

        # O(T·g) storage: proof + state per round
        proof_bytes = self.num_rounds * 192
        state_bytes = self.num_rounds * self.num_params * 4
        storage_mb = (proof_bytes + state_bytes) / (1024 ** 2)

        return {
            "scheme": "VeriCNN",
            "init_time_s": self.init_time,
            "single_prove_time_s": float(np.mean(self.prove_times)),
            "total_prove_time_s": float(np.sum(self.prove_times)),
            "verify_time_s": float(np.sum(self.verify_times)),
            "storage_mb": storage_mb,
        }
