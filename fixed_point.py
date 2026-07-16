"""
Fixed-point encoding utilities for zkPMT.

All real-valued data (weights, gradients, learning rate, training data)
are mapped to a finite field F_p using the complementary-code scheme
described in the paper.

  tilde_r = floor(r * 2^f)          if r >= 0
  tilde_r = p - floor(|r| * 2^f)    if r <  0

Scaling is delayed within each layer; rounding is applied once at the
end of each layer to avoid frequent gate operations.
"""

import numpy as np

# BLS12-381 scalar field prime (128-bit security)
P = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001

# Default fractional precision bits
DEFAULT_F = 16


def to_fixed_point(r: np.ndarray, f: int = DEFAULT_F, p: int = P) -> np.ndarray:
    """
    Convert a real-valued numpy array to fixed-point integers in F_p.

    Parameters
    ----------
    r : np.ndarray  (float)
    f : int         fractional precision bits
    p : int         field prime

    Returns
    -------
    np.ndarray of Python ints (or object dtype for large p)
    """
    scale = 2 ** f
    r_flat = r.flatten().astype(float)
    result = np.empty(len(r_flat), dtype=object)
    for i, val in enumerate(r_flat):
        if val >= 0:
            result[i] = int(np.floor(val * scale)) % p
        else:
            result[i] = (p - int(np.floor(abs(val) * scale))) % p
    return result.reshape(r.shape)


def from_fixed_point(tilde_r: np.ndarray, f: int = DEFAULT_F, p: int = P) -> np.ndarray:
    """
    Decode fixed-point integers back to approximate real values.
    Values > p//2 are treated as negative (complementary mapping).
    """
    scale = 2 ** f
    half_p = p // 2
    r_flat = tilde_r.flatten()
    result = np.empty(len(r_flat), dtype=float)
    for i, val in enumerate(r_flat):
        v = int(val)
        if v > half_p:
            result[i] = -(p - v) / scale
        else:
            result[i] = v / scale
    return result.reshape(tilde_r.shape)


def fixed_point_multiply(a: int, b: int, f: int = DEFAULT_F, p: int = P) -> int:
    """
    Multiply two fixed-point integers and rescale.
    tilde_z = floor(tilde_a * tilde_b / 2^f) mod p
    Scaling is verified inside the circuit; rounding is done outside.
    """
    product = (a * b) % p
    # Rescale: divide by 2^f (outside circuit, as per zkPMT design)
    return (product >> f) % p


def poseidon_hash(weights: np.ndarray, prev_hash: int = 0) -> int:
    """
    Lightweight simulation of the Poseidon hash function.

    In the real system this is a zk-friendly algebraic hash over F_p.
    Here we simulate it with a deterministic integer hash that captures
    the same chaining semantics:
        h_i = H(h_{i-1} || W_i)

    Parameters
    ----------
    weights   : np.ndarray  current model weights (fixed-point encoded)
    prev_hash : int         previous state hash h_{i-1}

    Returns
    -------
    int  new state hash h_i
    """
    # Flatten weights to a byte string for hashing
    import hashlib
    w_bytes = weights.astype(float).tobytes()
    prev_bytes = prev_hash.to_bytes(32, byteorder='big', signed=False)
    digest = hashlib.sha256(prev_bytes + w_bytes).digest()
    return int.from_bytes(digest, byteorder='big') % P


def encode_model_weights(model, f: int = DEFAULT_F) -> np.ndarray:
    """
    Extract all parameters from a PyTorch model and encode as fixed-point.
    Returns a 1-D numpy array of fixed-point integers.
    """
    import torch
    params = []
    for p_tensor in model.parameters():
        params.append(p_tensor.detach().cpu().numpy().flatten())
    all_params = np.concatenate(params)
    return to_fixed_point(all_params, f=f)
