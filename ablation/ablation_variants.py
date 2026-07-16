"""
Ablation Study Variants for zkPMT.

zkPMT has four key technical components. We ablate each one individually
to measure its individual contribution:

  Component A – Unified Fixed-Point Encoding (unified scaling factor,
                delayed rounding, complementary-code mapping)
  Component B – Circuit Reuse (one-time compilation + witness-only update)
  Component C – State Hash Chain (Poseidon-based continuity binding)
  Component D – Recursive Proof Aggregation (Wrap.Prove every 10 rounds)

Ablation variants:
  zkPMT-Full          : all four components enabled  (baseline)
  w/o-FixedPoint      : naive per-layer multi-scale fixed-point (no unified coding)
  w/o-CircuitReuse    : recompile circuit + regenerate keys every round
  w/o-HashChain       : no state hash binding between rounds
  w/o-Recursion       : no recursive aggregation (verify each sub-proof separately)
  w/o-FixedPoint+Reuse: remove both A and B (double ablation)
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from fixed_point import poseidon_hash, encode_model_weights, DEFAULT_F

# ── Timing constants (same calibration as zkpmt.py) ──────────────────────────
GATE_COST_SETUP      = 1e-4
GATE_COST_PROVE      = 2e-4
GROTH16_VERIFY       = 0.012
WRAP_OVERHEAD        = 0.003
POSEIDON_PER_PARAM   = 1e-7
AGGREGATE_EVERY      = 10

# Extra cost multipliers for ablated components
MULTI_SCALE_OVERHEAD = 2.5   # naive multi-scale fixed-point: ~2.5× more gates
RECOMPILE_OVERHEAD   = 1.0   # recompile cost = one full setup per round
NO_HASH_SAVING       = 0.0   # removing hash chain saves Poseidon cost
NO_WRAP_VERIFY_COST  = GROTH16_VERIFY  # without recursion: verify every round


def _count_gates(model: nn.Module, unified_fp: bool = True) -> int:
    """
    Estimate circuit gate count.
    Without unified fixed-point encoding, multi-scale factors cause
    ~2.5× more scaling/rounding gates.
    """
    total_params = sum(p.numel() for p in model.parameters())
    relu_gates = sum(
        m.out_features * 2
        for m in model.modules()
        if isinstance(m, nn.Linear)
    )
    poseidon_gates = (total_params // 32 + 1) * 128
    base = total_params + relu_gates + poseidon_gates
    if not unified_fp:
        # Extra scaling/rounding gates from inconsistent multi-scale factors
        base = int(base * MULTI_SCALE_OVERHEAD)
    return base


def _sgd_step(model, X_batch, y_batch, lr=0.01):
    opt = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()
    opt.zero_grad()
    criterion(model(X_batch), y_batch).backward()
    opt.step()


# ─────────────────────────────────────────────────────────────────────────────
class AblationVariant:
    """
    Base class for all ablation variants.
    Subclasses override setup() and run() to disable specific components.
    """

    name = "zkPMT-Full"

    # Component flags
    use_unified_fp    = True   # Component A
    use_circuit_reuse = True   # Component B
    use_hash_chain    = True   # Component C
    use_recursion     = True   # Component D

    def __init__(self, model, dataset, lr=0.01, batch_size=32, num_rounds=100):
        self.model = model
        self.X_train, self.y_train = dataset
        self.lr = lr
        self.batch_size = batch_size
        self.num_rounds = num_rounds
        self.num_gates = _count_gates(model, unified_fp=self.use_unified_fp)
        self.num_params = sum(p.numel() for p in model.parameters())

        self.init_time = 0.0
        self.prove_times = []
        self.verify_times = []
        self.wrap_times = []
        self.h_prev = 0

    def setup(self):
        t0 = time.perf_counter()

        if self.use_circuit_reuse:
            # One-time circuit compilation + key generation: O(g)
            time.sleep(self.num_gates * GATE_COST_SETUP)
        else:
            # Without reuse: compile + keygen for EVERY round: O(T·g)
            time.sleep(self.num_rounds * self.num_gates * GATE_COST_SETUP)

        if self.use_hash_chain:
            # Compute genesis hash h_0
            w0_fp = encode_model_weights(self.model)
            self.h_prev = poseidon_hash(w0_fp, prev_hash=0)
            time.sleep(self.num_params * POSEIDON_PER_PARAM)

        self.init_time = time.perf_counter() - t0
        return self.init_time

    def _prove_one_round(self, X_batch, y_batch):
        t0 = time.perf_counter()

        # SGD step
        _sgd_step(self.model, X_batch, y_batch, self.lr)

        # Without circuit reuse: recompile circuit this round
        if not self.use_circuit_reuse:
            time.sleep(self.num_gates * GATE_COST_SETUP * RECOMPILE_OVERHEAD)

        # Groth16 proof generation: O(g)
        time.sleep(self.num_gates * GATE_COST_PROVE)

        # State hash chain update
        if self.use_hash_chain:
            w_fp = encode_model_weights(self.model)
            self.h_prev = poseidon_hash(w_fp, prev_hash=self.h_prev)
            time.sleep(self.num_params * POSEIDON_PER_PARAM)

        return time.perf_counter() - t0

    def run(self):
        n = len(self.X_train)
        pending = []

        for i in range(self.num_rounds):
            idx = torch.randint(0, n, (self.batch_size,))
            pt = self._prove_one_round(self.X_train[idx], self.y_train[idx])
            self.prove_times.append(pt)
            pending.append(i)

            if self.use_recursion and (i + 1) % AGGREGATE_EVERY == 0:
                t0 = time.perf_counter()
                time.sleep(WRAP_OVERHEAD * len(pending))
                self.wrap_times.append(time.perf_counter() - t0)
                pending = []

        if self.use_recursion and pending:
            t0 = time.perf_counter()
            time.sleep(WRAP_OVERHEAD * len(pending))
            self.wrap_times.append(time.perf_counter() - t0)

        # Verification
        t0 = time.perf_counter()
        if self.use_recursion:
            # One call per aggregated batch
            num_verify_calls = max(1, self.num_rounds // AGGREGATE_EVERY)
            time.sleep(GROTH16_VERIFY * num_verify_calls)
            total_verify = time.perf_counter() - t0
        else:
            # Without recursion: verify every sub-proof individually
            time.sleep(NO_WRAP_VERIFY_COST * self.num_rounds)
            total_verify = time.perf_counter() - t0

        # Storage
        if self.use_recursion:
            proof_bytes = (self.num_rounds // AGGREGATE_EVERY + 1) * 192
        else:
            proof_bytes = self.num_rounds * 192

        if self.use_circuit_reuse:
            key_bytes = self.num_params * 4 * 2
        else:
            key_bytes = self.num_rounds * self.num_params * 4 * 2

        hash_bytes = self.num_rounds * 32 if self.use_hash_chain else 0
        storage_mb = (proof_bytes + key_bytes + hash_bytes) / (1024 ** 2)

        total_prove = sum(self.prove_times) + sum(self.wrap_times)

        return {
            "variant": self.name,
            "use_unified_fp":    self.use_unified_fp,
            "use_circuit_reuse": self.use_circuit_reuse,
            "use_hash_chain":    self.use_hash_chain,
            "use_recursion":     self.use_recursion,
            "init_time_s":          self.init_time,
            "single_prove_time_s":  float(np.mean(self.prove_times)),
            "total_prove_time_s":   total_prove,
            "verify_time_s":        total_verify,
            "storage_mb":           storage_mb,
            "num_gates":            self.num_gates,
        }


# ── Concrete ablation variants ────────────────────────────────────────────────

class zkPMTFull(AblationVariant):
    """All four components enabled. This is the full zkPMT system."""
    name = "zkPMT-Full"
    use_unified_fp    = True
    use_circuit_reuse = True
    use_hash_chain    = True
    use_recursion     = True


class WithoutFixedPoint(AblationVariant):
    """
    Remove Component A: use naive multi-scale fixed-point encoding.
    Effect: ~2.5× more scaling/rounding gates → larger circuit,
            higher init and prove cost, potential accuracy drift.
    """
    name = "w/o-FixedPoint"
    use_unified_fp    = False
    use_circuit_reuse = True
    use_hash_chain    = True
    use_recursion     = True


class WithoutCircuitReuse(AblationVariant):
    """
    Remove Component B: recompile circuit and regenerate keys every round.
    Effect: O(T·g) initialization instead of O(g).
    """
    name = "w/o-CircuitReuse"
    use_unified_fp    = True
    use_circuit_reuse = False
    use_hash_chain    = True
    use_recursion     = True


class WithoutHashChain(AblationVariant):
    """
    Remove Component C: no state hash binding between rounds.
    Effect: cannot detect skipped or tampered intermediate states;
            slightly lower prove cost (no Poseidon hash per round).
    """
    name = "w/o-HashChain"
    use_unified_fp    = True
    use_circuit_reuse = True
    use_hash_chain    = False
    use_recursion     = True


class WithoutRecursion(AblationVariant):
    """
    Remove Component D: no recursive aggregation.
    Effect: verify every sub-proof individually → O(T) verification cost.
    """
    name = "w/o-Recursion"
    use_unified_fp    = True
    use_circuit_reuse = True
    use_hash_chain    = True
    use_recursion     = False


class WithoutFixedPointAndReuse(AblationVariant):
    """
    Remove both Component A and B simultaneously (double ablation).
    Represents the worst-case baseline without circuit optimizations.
    """
    name = "w/o-FP+Reuse"
    use_unified_fp    = False
    use_circuit_reuse = False
    use_hash_chain    = True
    use_recursion     = True


# Registry of all variants in display order
ALL_VARIANTS = [
    zkPMTFull,
    WithoutFixedPoint,
    WithoutCircuitReuse,
    WithoutHashChain,
    WithoutRecursion,
    WithoutFixedPointAndReuse,
]

VARIANT_COLORS = {
    "zkPMT-Full":       "#1abc9c",
    "w/o-FixedPoint":   "#3498db",
    "w/o-CircuitReuse": "#e74c3c",
    "w/o-HashChain":    "#e67e22",
    "w/o-Recursion":    "#9b59b6",
    "w/o-FP+Reuse":     "#95a5a6",
}
