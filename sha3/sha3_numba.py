"""SHA3 implementation in Python in functional style with Numba acceleration."""

import numpy as np
from numba import njit

_KECCAK_RHO = np.array([
     1,  3,  6, 10, 15, 21,
    28, 36, 45, 55,  2, 14,
    27, 41, 56,  8, 25, 43,
    62, 18, 39, 61, 20, 44],
    dtype=np.uint64)
_KECCAK_PI = np.array([
    10,  7, 11, 17, 18, 3,
     5, 16,  8, 21, 24, 4,
    15, 23, 19, 13, 12, 2,
    20, 14, 22,  9,  6, 1],
    dtype=np.uint64)
_KECCAK_RC = np.array([
  0x0000000000000001, 0x0000000000008082,
  0x800000000000808a, 0x8000000080008000,
  0x000000000000808b, 0x0000000080000001,
  0x8000000080008081, 0x8000000000008009,
  0x000000000000008a, 0x0000000000000088,
  0x0000000080008009, 0x000000008000000a,
  0x000000008000808b, 0x800000000000008b,
  0x8000000000008089, 0x8000000000008003,
  0x8000000000008002, 0x8000000000000080,
  0x000000000000800a, 0x800000008000000a,
  0x8000000080008081, 0x8000000000008080,
  0x0000000080000001, 0x8000000080008008],
  dtype=np.uint64)

_DSBYTE = 0x06

@njit
def _rol(x, s):
    """Rotate x left by s."""
    return ((np.uint64(x) << np.uint64(s)) ^ (np.uint64(x) >> np.uint64(64 - s)))

@njit
def _keccak_f(state):
    """
    The keccak_f permutation function.

    """

    bc = np.zeros(5, dtype=np.uint64)

    for i in range(24):
        # Parity
        for x in range(5):
            bc[x] = 0
            for y in range(0, 25, 5):
                bc[x] ^= state[x + y]

        # Theta
        for x in range(5):
            t = bc[(x + 4) % 5] ^ _rol(bc[(x + 1) % 5], 1)
            for y in range(0, 25, 5):
                state[y + x] ^= t

        # Rho and pi
        t = state[1]
        for x in range(24):
            bc[0] = state[_KECCAK_PI[x]]
            state[_KECCAK_PI[x]] = _rol(t, _KECCAK_RHO[x])
            t = bc[0]

        for y in range(0, 25, 5):
            for x in range(5):
                bc[x] = state[y + x]
            for x in range(5):
                state[y + x] = bc[x] ^ ((~bc[(x + 1) % 5]) & bc[(x + 2) % 5])

        state[0] ^= _KECCAK_RC[i]

    return state

@njit
def _absorb(state, rate, buf, buf_idx, b):
    """
    Absorb input data into the sponge construction.

    Args:
        b (bytes): The input data to be absorbed.

    """
    todo = len(b)
    i = 0
    while todo > 0:
        cando = rate - buf_idx
        willabsorb = min(cando, todo)
        buf[buf_idx:buf_idx + willabsorb] ^= \
            np.frombuffer(b[i:i+willabsorb], dtype=np.uint8)
        buf_idx += willabsorb
        if buf_idx == rate:
            state, buf, buf_idx = _permute(state, buf, buf_idx)
        todo -= willabsorb
        i += willabsorb

    return state, buf, buf_idx

@njit
def _squeeze(state, rate, buf, buf_idx, n):
    """
    Squeeze output data from the sponge construction.
    """
    tosqueeze = n
    output_bytes = np.empty(tosqueeze, dtype=np.uint8)  # Temporary storage for output bytes
    output_index = 0  # Tracks where to insert bytes into output_bytes

    while tosqueeze > 0:
        cansqueeze = rate - buf_idx
        willsqueeze = min(cansqueeze, tosqueeze)

        # Extract bytes from state
        for _ in range(willsqueeze):
            byte_index = buf_idx % 8
            byte_val = (state[buf_idx // 8] >> (byte_index * 8)) & 0xFF
            output_bytes[output_index] = byte_val
            buf_idx += 1
            output_index += 1

            # If we've processed a full rate's worth of data, permute
            if buf_idx == rate:
                state, buf, buf_idx = _permute(state, buf, 0)  # Reset buf_idx for simplicity

        tosqueeze -= willsqueeze

    return output_bytes

@njit
def _pad(state, rate, buf, buf_idx):
    """
    Pad the input data in the buffer.

    """
    buf[buf_idx] ^= _DSBYTE
    buf[rate - 1] ^= 0x80
    return _permute(state, buf, buf_idx)

@njit
def _permute(state, buf, buf_idx):
    """
    Permute the internal state and buffer for thorough mixing.
    Uses a manual workaround since numba doesn't support np.view.
    """
    temp_state = np.zeros(len(state), dtype=np.uint64)

    # Process bytes to uint64
    for i in range(0, len(buf), 8):
        if i + 8 <= len(buf):  # Ensure there's enough data to read
            uint64_val = np.uint64(0)
            for j in range(8):
                uint64_val |= np.uint64(buf[i+j]) << (j * 8)
            temp_state[i//8] = uint64_val

    state ^= temp_state

    # Perform Keccak permutation
    state = _keccak_f(state)

    # Reset buf_idx and buf
    buf_idx = 0
    buf[:] = 0

    return state, buf, buf_idx

@njit
def sha3(bit_length, data=b''):
    """
    Compute the SHA-3 hash of the input data.

    Args:
        bit_length (int): The bit length of the hash.
        data (bytes): The input data to hash.

    Returns:
        bytes (np.ndarray): A uint8 array of the hash.

    """
    rate_map = {224: 144, 256: 136, 384: 104, 512: 72}
    if bit_length not in rate_map:
        raise ValueError('Invalid bit length.')

    rate =  rate_map[bit_length]
    buf_idx = 0
    state = np.zeros(25, dtype=np.uint64)
    buf = np.zeros(200, dtype=np.uint8) # confirm this is correct

    # Absorb data
    state, buf, buf_idx = _absorb(state, rate, buf, buf_idx, data)

    # Pad in preparation for squeezing
    state, buf, buf_idx = _pad(state, rate, buf, buf_idx)

    # Squeeze the hash
    return _squeeze(state, rate, buf, buf_idx, bit_length // 8)