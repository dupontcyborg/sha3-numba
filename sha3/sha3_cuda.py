"""SHA3 implementation in Python in functional style with Numba acceleration."""

import numpy as np
from numba import cuda, uint64, uint8

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

@cuda.jit(device=True)
def _rol(x, s):
    """Rotate x left by s."""
    return ((np.uint64(x) << np.uint64(s)) ^ (np.uint64(x) >> np.uint64(64 - s)))

@cuda.jit(device=True)
def _keccak_f(state):
    """
    The keccak_f permutation function, unrolled for performance.

    """

    bc = cuda.local.array(5, dtype=uint64)
    for i in range(25):
        bc[i] = 0

    for i in range(24):

        # Parity calculation unrolled
        bc[0] = state[0] ^ state[5] ^ state[10] ^ state[15] ^ state[20]
        bc[1] = state[1] ^ state[6] ^ state[11] ^ state[16] ^ state[21]
        bc[2] = state[2] ^ state[7] ^ state[12] ^ state[17] ^ state[22]
        bc[3] = state[3] ^ state[8] ^ state[13] ^ state[18] ^ state[23]
        bc[4] = state[4] ^ state[9] ^ state[14] ^ state[19] ^ state[24]

        # Theta unrolled
        t0 = bc[4] ^ _rol(bc[1], 1)
        t1 = bc[0] ^ _rol(bc[2], 1)
        t2 = bc[1] ^ _rol(bc[3], 1)
        t3 = bc[2] ^ _rol(bc[4], 1)
        t4 = bc[3] ^ _rol(bc[0], 1)

        state[0] ^= t0
        state[5] ^= t0
        state[10] ^= t0
        state[15] ^= t0
        state[20] ^= t0

        state[1] ^= t1
        state[6] ^= t1
        state[11] ^= t1
        state[16] ^= t1
        state[21] ^= t1

        state[2] ^= t2
        state[7] ^= t2
        state[12] ^= t2
        state[17] ^= t2
        state[22] ^= t2

        state[3] ^= t3
        state[8] ^= t3
        state[13] ^= t3
        state[18] ^= t3
        state[23] ^= t3

        state[4] ^= t4
        state[9] ^= t4
        state[14] ^= t4
        state[19] ^= t4
        state[24] ^= t4

        # Rho and Pi unrolled
        t1  = _rol(state[1], 1)
        t2  = _rol(state[10], 3)
        t3  = _rol(state[7], 6)
        t4  = _rol(state[11], 10)
        t5  = _rol(state[17], 15)
        t6  = _rol(state[18], 21)
        t7  = _rol(state[3], 28)
        t8  = _rol(state[5], 36)
        t9  = _rol(state[16], 45)
        t10 = _rol(state[8], 55)
        t11 = _rol(state[21], 2)
        t12 = _rol(state[24], 14)
        t13 = _rol(state[4], 27)
        t14 = _rol(state[15], 41)
        t15 = _rol(state[23], 56)
        t16 = _rol(state[19], 8)
        t17 = _rol(state[13], 25)
        t18 = _rol(state[12], 43)
        t19 = _rol(state[2], 62)
        t20 = _rol(state[20], 18)
        t21 = _rol(state[14], 39)
        t22 = _rol(state[22], 61)
        t23 = _rol(state[9], 20)
        t24 = _rol(state[6], 44)

        state[10] = t1
        state[7]  = t2
        state[11] = t3
        state[17] = t4
        state[18] = t5
        state[3]  = t6
        state[5]  = t7
        state[16] = t8
        state[8]  = t9
        state[21] = t10
        state[24] = t11
        state[4]  = t12
        state[15] = t13
        state[23] = t14
        state[19] = t15
        state[13] = t16
        state[12] = t17
        state[2]  = t18
        state[20] = t19
        state[14] = t20
        state[22] = t21
        state[9]  = t22
        state[6]  = t23
        state[1]  = t24

        # Chi unrolled
        t0  = state[0] ^ ((~state[1]) & state[2])
        t1  = state[1] ^ ((~state[2]) & state[3])
        t2  = state[2] ^ ((~state[3]) & state[4])
        t3  = state[3] ^ ((~state[4]) & state[0])
        t4  = state[4] ^ ((~state[0]) & state[1])
        t5  = state[5] ^ ((~state[6]) & state[7])
        t6  = state[6] ^ ((~state[7]) & state[8])
        t7  = state[7] ^ ((~state[8]) & state[9])
        t8  = state[8] ^ ((~state[9]) & state[5])
        t9  = state[9] ^ ((~state[5]) & state[6])
        t10 = state[10] ^ ((~state[11]) & state[12])
        t11 = state[11] ^ ((~state[12]) & state[13])
        t12 = state[12] ^ ((~state[13]) & state[14])
        t13 = state[13] ^ ((~state[14]) & state[10])
        t14 = state[14] ^ ((~state[10]) & state[11])
        t15 = state[15] ^ ((~state[16]) & state[17])
        t16 = state[16] ^ ((~state[17]) & state[18])
        t17 = state[17] ^ ((~state[18]) & state[19])
        t18 = state[18] ^ ((~state[19]) & state[15])
        t19 = state[19] ^ ((~state[15]) & state[16])
        t20 = state[20] ^ ((~state[21]) & state[22])
        t21 = state[21] ^ ((~state[22]) & state[23])
        t22 = state[22] ^ ((~state[23]) & state[24])
        t23 = state[23] ^ ((~state[24]) & state[20])
        t24 = state[24] ^ ((~state[20]) & state[21])

        state[0]  = t0
        state[1]  = t1
        state[2]  = t2
        state[3]  = t3
        state[4]  = t4
        state[5]  = t5
        state[6]  = t6
        state[7]  = t7
        state[8]  = t8
        state[9]  = t9
        state[10] = t10
        state[11] = t11
        state[12] = t12
        state[13] = t13
        state[14] = t14
        state[15] = t15
        state[16] = t16
        state[17] = t17
        state[18] = t18
        state[19] = t19
        state[20] = t20
        state[21] = t21
        state[22] = t22
        state[23] = t23
        state[24] = t24

        state[0] ^= _KECCAK_RC[i]

    return state

@cuda.jit(device=True)
def _absorb(state, rate, buf, buf_idx, b):
    """
    Absorb input data into the sponge construction in a CUDA-friendly way.

    Args:
        state: The state array of the SHA-3 sponge construction.
        rate: The rate of the sponge function.
        buf: The buffer to absorb the input into.
        buf_idx: Current index in the buffer.
        b: The input data to be absorbed, expected to be a device array.
    """
    todo = len(b)
    i = 0
    while todo > 0:
        cando = rate - buf_idx
        willabsorb = min(cando, todo)
        for j in range(willabsorb):
            # Directly manipulate each byte rather than using numpy operations
            buf[buf_idx + j] ^= b[i + j]
        buf_idx += willabsorb
        if buf_idx == rate:
            state, buf, buf_idx = _permute(state, buf, buf_idx)  # Ensure _permute is also device-friendly
        todo -= willabsorb
        i += willabsorb

    return state, buf, buf_idx

@cuda.jit(device=True)
def _squeeze(state, bit_length, rate, buf, buf_idx, output_buf, output_idx):
    """
    Directly updates output_buf in-place.
    """

    tosqueeze = bit_length // 8
    local_output_idx = 0  # Tracks where to insert bytes into output_buf

    while tosqueeze > 0:
        cansqueeze = rate - buf_idx
        willsqueeze = min(cansqueeze, tosqueeze)

        # Extract bytes from state and directly update output_buf
        for _ in range(willsqueeze):
            byte_index = buf_idx % 8
            byte_val = (state[buf_idx // 8] >> (byte_index * 8)) & 0xFF

            output_buf[output_idx, local_output_idx] = byte_val

            buf_idx += 1
            local_output_idx += 1

            # If we've processed a full rate's worth of data, permute
            if buf_idx == rate:
                state, buf, buf_idx = _permute(state, buf, 0)

        tosqueeze -= willsqueeze


@cuda.jit(device=True)
def _pad(state, rate, buf, buf_idx):
    """
    Pad the input data in the buffer.

    """
    buf[buf_idx] ^= _DSBYTE
    buf[rate - 1] ^= 0x80
    return _permute(state, buf, buf_idx)

@cuda.jit(device=True)
def _permute(state, buf, buf_idx):
    """
    Permute the internal state and buffer for thorough mixing.
    Uses a manual workaround since Numba doesn't support np.view.
    """
    temp_state = cuda.local.array(25, dtype=uint64)
    for i in range(25):
        temp_state[i] = 0

    # Process bytes to uint64
    for i in range(0, len(buf), 8):
        if i + 8 <= len(buf):  # Ensure there's enough data to read
            uint64_val = uint64(0)
            for j in range(8):
                uint64_val |= uint64(buf[i+j]) << (j * 8)
            temp_state[i//8] = uint64_val

    # Manually perform bitwise XOR for each element
    for i in range(25):
        state[i] ^= temp_state[i]

    # Perform Keccak permutation
    state = _keccak_f(state)

    # Reset buf_idx and buf
    buf_idx = 0
    for i in range(200):
        buf[i] = 0

    return state, buf, buf_idx


@cuda.jit()
def sha3(bit_length, data_gpu, output_gpu, num_iterations=1):
    """
    Compute the SHA-3 hash of the input data.

    Args:
        bit_length (int): The bit length of the hash.
        data (bytes): The input data to hash.

    Returns:
        bytes (np.ndarray): A uint8 array of the hash.

    """

    idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    if idx < num_iterations:
        # Implement the logic to compute hash for each thread.
        # Each thread can call sha3 with its own data segment or modifications.
        pass

    if bit_length not in (224, 256, 384, 512):
        raise ValueError('Invalid bit length.')
    
    # Compute rate
    rate = 200 - (bit_length // 4)

    buf_idx = 0

    state = cuda.local.array(25, dtype=uint64)
    buf = cuda.local.array(200, dtype=uint8)
    
    # Reset state and buf for demonstration purposes
    for i in range(25):
        state[i] = 0
    for i in range(200):
        buf[i] = 0

    # Absorb data
    state, buf, buf_idx = _absorb(state, rate, buf, buf_idx, data_gpu)

    # Pad in preparation for squeezing
    state, buf, buf_idx = _pad(state, rate, buf, buf_idx)

    # Squeeze the hash based on desired bit length
    _squeeze(state, bit_length, rate, buf, buf_idx, output_gpu, idx)

def test_sha3_cuda(bit_length, data=b'', num_iterations=100):
    """
    Testing function for the CUDA-accelerated SHA-3 implementation.
    """

    rate_map = {224: 144, 256: 136, 384: 104, 512: 72}
    if bit_length not in rate_map:
        raise ValueError('Invalid bit length.')

    # Define kernel execution configuration
    threads_per_block = 128
    blocks_per_grid = (num_iterations + threads_per_block - 1) // threads_per_block

    data = np.array(list(data), dtype=np.uint8)

    # Allocate memory on the device
    data_gpu = cuda.to_device(data)
    output_gpu = cuda.device_array((num_iterations, bit_length // 8), dtype=np.uint8)

    # Launch the kernel
    sha3[blocks_per_grid, threads_per_block](bit_length, data_gpu, output_gpu, num_iterations)

    # Copy the result back to host memory
    output = output_gpu.copy_to_host()

    return output
