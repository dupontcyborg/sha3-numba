"""SHA3 implementation in Python in functional style"""

import numpy as np

_KECCAK_RHO = [
     1,  3,  6, 10, 15, 21,
    28, 36, 45, 55,  2, 14,
    27, 41, 56,  8, 25, 43,
    62, 18, 39, 61, 20, 44]
_KECCAK_PI = [
    10,  7, 11, 17, 18, 3,
     5, 16,  8, 21, 24, 4,
    15, 23, 19, 13, 12, 2,
    20, 14, 22,  9,  6, 1]
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

_SPONGE_ABSORBING = 1
_SPONGE_SQUEEZING = 2
_DSBYTE = 0x06

def _rol(x, s):
    """Rotate x left by s."""
    return ((np.uint64(x) << np.uint64(s)) ^ (np.uint64(x) >> np.uint64(64 - s)))

def _keccak_f(ctx):
    """
    The keccak_f permutation function.

    """
    state = ctx['state']
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
    ctx['state'] = state

def _absorb(ctx, b):
    """
    Absorb input data into the sponge construction.

    Args:
        b (bytes): The input data to be absorbed.

    """
    todo = len(b)
    i = 0
    while todo > 0:
        cando = ctx['rate'] - ctx['i']
        willabsorb = min(cando, todo)
        ctx['buf'][ctx['i']:ctx['i'] + willabsorb] ^= \
            np.frombuffer(b[i:i+willabsorb], dtype=np.uint8)
        ctx['i'] += willabsorb
        if ctx['i'] == ctx['rate']:
            _permute(ctx)
        todo -= willabsorb
        i += willabsorb

def _squeeze(ctx, n):
    """
    Squeeze output data from the sponge construction.

    Args:
        n (int): The number of bytes to squeeze.

    Returns:
        bytes: The squeezed output data.

    """
    tosqueeze = n
    b = b''
    while tosqueeze > 0:
        cansqueeze = ctx['rate'] - ctx['i']
        willsqueeze = min(cansqueeze, tosqueeze)
        b += ctx['state'].view(dtype=np.uint8)[ctx['i']:ctx['i'] + willsqueeze].tostring()
        ctx['i'] += willsqueeze
        if ctx['i'] == ctx['rate']:
            _permute(ctx)
        tosqueeze -= willsqueeze
    return b

def _pad(ctx):
    """
    Pad the input data in the buffer.

    """
    ctx['buf'][ctx['i']] ^= _DSBYTE
    ctx['buf'][ctx['rate'] - 1] ^= 0x80
    _permute(ctx)

def _permute(ctx):
    """
    Permute the internal state and buffer.

    """
    ctx['state'] ^= ctx['buf'].view(dtype=np.uint64)
    _keccak_f(ctx)
    ctx['i'] = 0
    ctx['buf'][:] = 0

def sha3(bit_length, data=b''):
    """
    Compute the SHA-3 hash of the input data.

    Args:
        bit_length (int): The bit length of the hash.
        data (bytes): The input data to hash.

    Returns:
        bytes: The hash of the input data.

    """
    rate_map = {224: 144, 256: 136, 384: 104, 512: 72}
    if bit_length not in rate_map:
        raise ValueError('Invalid bit length.')

    # Setup context
    ctx = {
        'bit_length': bit_length,
        'rate': rate_map[bit_length],
        'i': 0,
        'state': np.zeros(25, dtype=np.uint64),
        'buf': np.zeros(200, dtype=np.uint8), # confirm this is correct
    }

    # Absorb data
    _absorb(ctx, data)

    # Pad in preparation for squeezing
    _pad(ctx)

    # Squeeze the hash
    return _squeeze(ctx, bit_length // 8)