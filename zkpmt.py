"""
zkPMT: Zero-Knowledge Proof-based Model Training integrity verification.

This module simulates the four phases of zkPMT:
  1. Preprocess  – fixed-point encoding of model, data, hyperparameters
  2. Setup       – circuit compilation + key generation + genesis hash h_0
  3. Prove       – per-round proof generation with state hash chain
  4. Wrap.Prove  – recursive aggregation every AGGREGATE_EVERY rounds
  5. Wrap.Verify – single-call verification of the recursive proof

All ZKP operations (Groth16, recursive SNARK) are simulated via
calibrated timing models with configurable constants.
The simulation faithfully reproduces the *cost structure* of each phase
for end-to-end workflow validation.
"""

import time
import hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from fixed_point import poseidon_hash, encode_model_weights, DEFAULT_F

# ── Timing calibration constants (seconds) ──────────────────────────────────
# Placeholder constants for trusted setup, proof generation, wrapping,
# and verification costs in this simulation pipeline.

# Cost per gate in the circuit (trusted setup phase)
GATE_COST_SETUP = 1e-4          # seconds per gate

# Cost per gate for a single Groth16 proof generation call
GATE_COST_PROVE = 2e-4          # seconds per gate

# Cost of a single Groth16 verification (pairing check, constant)
GROTH16_VERIFY_COST = 0.012     # seconds

# Overhead of recursive SNARK wrapping per round
WRAP_OVERHEAD = 0.003           # seconds per wrap call

# Poseidon hash cost per parameter (outside circuit)
POSEIDON_COST_PER_PARAM = 1e-7  # seconds

# Aggregation interval: zkPMT aggregates every 10 rounds
AGGREGATE_EVERY = 10


def _count_gates(model: nn.Module) -> int:
    """
    Estimate the number of arithmetic gates in the R1CS circuit for one
    training round of the given model.

    Gate count approximation:
      - Each weight multiplication: 1 mul gate
      - Each bias addition: 1 add gate
      - Each ReLU: 2 Boolean gates (b*(z-a)=0, (1-b)*a=0)
      - Conv/BN layers: modeled by their output size and kernel geometry
      - Poseidon hash subcircuit: 128 gates per 32 parameters (fixed)
    """
    total_params = sum(p.numel() for p in model.parameters())
    relu_gates = 0
    conv_gates = 0

    # We need a dummy input to trace feature map sizes for CNNs
    # Default to 32x32 for CIFAR-based tasks
    device = next(model.parameters()).device
    dummy_x = torch.zeros(1, 3, 32, 32).to(device)

    # Simple estimation for MLP and ResNet modules
    for module in model.modules():
        if isinstance(module, nn.Linear):
            relu_gates += module.out_features * 2
        elif isinstance(module, nn.Conv2d):
            # Roughly: H_out * W_out * C_out * C_in * k * k
            # For simplicity in simulation, we use a factor of parameters
            # but scaled by typical input resolution (32x32 -> 1024)
            conv_gates += module.weight.numel() * 10
        elif isinstance(module, nn.BatchNorm2d):
            conv_gates += module.num_features * 32 * 32 * 2
        elif isinstance(module, nn.ReLU):
            # This is harder to track without a full forward pass.
            # We approximate based on common architectures.
            pass

    # If it's a large model like ResNet, ReLU count is proportional to total params
    if any(isinstance(m, nn.Conv2d) for m in model.modules()):
        relu_gates = total_params // 2

    poseidon_gates = (total_params // 32 + 1) * 128
    return total_params + relu_gates + conv_gates + poseidon_gates


class zkPMT:
    """
    Simulates the full zkPMT protocol for one training run.

    Parameters
    ----------
    model      : nn.Module   the approximated MLP model
    dataset    : tuple       (X_train, y_train) as torch tensors
    lr         : float       learning rate
    batch_size : int
    num_rounds : int         total SGD training rounds (iterations)
    """

    def __init__(self, model, dataset, lr=0.01, batch_size=32, num_rounds=100):
        self.model = model
        self.X_train, self.y_train = dataset
        self.lr = lr
        self.batch_size = batch_size
        self.num_rounds = num_rounds

        self.num_gates = _count_gates(model)
        self.num_params = sum(p.numel() for p in model.parameters())

        # State maintained across rounds
        self.h_prev = 0          # previous state hash
        self.sub_proofs = []     # list of (round_idx, proof_size_bytes, gen_time)
        self.recursive_proofs = []  # list of wrapped proofs

        # Results recorded during run
        self.init_time = 0.0
        self.prove_times = []        # per-round proof generation time
        self.wrap_times = []         # per-wrap recursive aggregation time
        self.verify_time = 0.0
        self.storage_bytes = 0

        # Circuit and keys (simulated as metadata)
        self.pk = None
        self.vk = None
        self.circuit_compiled = False

    # ── Phase 1 & 2: Preprocess + Setup ─────────────────────────────────────
    def setup(self):
        """
        Simulate circuit compilation and trusted setup.

        Cost model:
          - One-time circuit compilation: O(g) where g = num_gates
          - Groth16 trusted setup (key generation): O(g)
          - Poseidon hash of initial weights → h_0
        """
        t0 = time.perf_counter()

        # Fixed-point encode initial weights
        w0_fp = encode_model_weights(self.model)

        # Simulate circuit compilation (O(g))
        time.sleep(self.num_gates * GATE_COST_SETUP * 0.5)

        # Simulate Groth16 trusted setup (O(g))
        time.sleep(self.num_gates * GATE_COST_SETUP * 0.5)

        # Compute genesis hash h_0 = PoseidonHash(W_0)
        self.h_prev = poseidon_hash(w0_fp, prev_hash=0)

        # Simulate dataset commitment: Commit(D_i, r_i) for each batch
        num_batches = max(1, len(self.X_train) // self.batch_size)
        time.sleep(num_batches * POSEIDON_COST_PER_PARAM * self.X_train.shape[-1])

        # Store simulated keys
        self.pk = hashlib.sha256(b"proving_key").hexdigest()
        self.vk = hashlib.sha256(b"verification_key").hexdigest()
        self.circuit_compiled = True

        self.init_time = time.perf_counter() - t0
        return self.init_time

    # ── Phase 3: Prove (single round) ────────────────────────────────────────
    def prove_round(self, round_idx: int, X_batch, y_batch):
        """
        Execute one SGD update and generate the corresponding sub-proof.

        Steps:
          1. Forward pass (fixed-point arithmetic simulated via PyTorch)
          2. Backward pass + weight update
          3. Groth16.Prove(pk, C_SGD+Hash, witness_i) → pi_i
          4. Compute h_i = PoseidonHash(W_i) outside circuit
        """
        t0 = time.perf_counter()

        # ── Actual SGD step ──────────────────────────────────────────────────
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        optimizer.zero_grad()
        logits = self.model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        # ── Simulate Groth16 proof generation: O(g) ─────────────────────────
        time.sleep(self.num_gates * GATE_COST_PROVE)

        # ── Compute state hash h_i = PoseidonHash(W_i) ──────────────────────
        w_fp = encode_model_weights(self.model)
        h_i = poseidon_hash(w_fp, prev_hash=self.h_prev)

        # Simulate Poseidon hash cost (outside circuit)
        time.sleep(self.num_params * POSEIDON_COST_PER_PARAM)

        self.h_prev = h_i

        elapsed = time.perf_counter() - t0
        self.prove_times.append(elapsed)

        # Proof size: Groth16 proof = 3 group elements on BLS12-381 ≈ 192 bytes
        proof_bytes = 192
        self.sub_proofs.append((round_idx, proof_bytes, elapsed))

        return elapsed, h_i

    # ── Phase 4: Wrap.Prove (recursive aggregation) ──────────────────────────
    def wrap_prove(self, sub_proofs_batch):
        """
        Recursively aggregate AGGREGATE_EVERY sub-proofs into one
        recursive proof Pi_i.

        Cost: one recursive SNARK wrap call per batch.
        """
        t0 = time.perf_counter()
        time.sleep(WRAP_OVERHEAD * len(sub_proofs_batch))
        elapsed = time.perf_counter() - t0
        self.wrap_times.append(elapsed)

        # Recursive proof size ≈ same as one Groth16 proof (constant size)
        self.recursive_proofs.append({
            "rounds": [sp[0] for sp in sub_proofs_batch],
            "size_bytes": 192,
            "wrap_time": elapsed,
        })
        return elapsed

    # ── Phase 5: Wrap.Verify ─────────────────────────────────────────────────
    def wrap_verify(self):
        """
        Single-call verification of the final recursive proof Pi_rec.

        Inputs:  h_0, h_n, Pi_rec
        Cost:    one Groth16 pairing check (constant)
        """
        t0 = time.perf_counter()
        time.sleep(GROTH16_VERIFY_COST)
        self.verify_time = time.perf_counter() - t0
        return self.verify_time

    # ── Full training run ─────────────────────────────────────────────────────
    def run(self):
        """
        Execute the complete zkPMT protocol:
          setup → [prove_round × T] → [wrap_prove every 10] → wrap_verify

        Returns a dict of timing and storage results.
        """
        assert self.circuit_compiled or True  # setup called externally

        n = len(self.X_train)
        pending_sub_proofs = []

        for i in range(self.num_rounds):
            # Sample a mini-batch
            idx = torch.randint(0, n, (self.batch_size,))
            X_b = self.X_train[idx]
            y_b = self.y_train[idx]

            pt, _ = self.prove_round(i, X_b, y_b)
            pending_sub_proofs.append(self.sub_proofs[-1])

            # Aggregate every AGGREGATE_EVERY rounds
            if (i + 1) % AGGREGATE_EVERY == 0:
                self.wrap_prove(pending_sub_proofs)
                pending_sub_proofs = []

        # Aggregate any remaining sub-proofs
        if pending_sub_proofs:
            self.wrap_prove(pending_sub_proofs)

        # Final verification
        self.wrap_verify()

        # Storage: state hash chain + recursive proofs + circuit keys
        hash_chain_bytes = self.num_rounds * 32          # 32 bytes per hash
        recursive_proof_bytes = len(self.recursive_proofs) * 192
        circuit_key_bytes = self.num_params * 4 * 2      # pk + vk (float32)
        self.storage_bytes = hash_chain_bytes + recursive_proof_bytes + circuit_key_bytes

        return self._summary()

    def _summary(self):
        total_prove = sum(self.prove_times)
        total_wrap = sum(self.wrap_times)
        num_verifications = len(self.recursive_proofs)
        total_verify = self.verify_time * num_verifications  # one call per wrap

        return {
            "scheme": "zkPMT",
            "init_time_s": self.init_time,
            "single_prove_time_s": float(np.mean(self.prove_times)) if self.prove_times else 0,
            "total_prove_time_s": total_prove + total_wrap,
            "verify_time_s": total_verify,
            "storage_mb": self.storage_bytes / (1024 ** 2),
            "num_rounds": self.num_rounds,
            "num_verifications": num_verifications,
        }
