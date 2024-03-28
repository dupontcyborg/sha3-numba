import numpy as np
from numba import jit

# Number of rounds for Keccak-f
NUM_ROUNDS = 24

# Define the round constants for Keccak-f
ROUND_CONSTANTS = np.array([
    0x0000000000000001, 0x0000000000008082, 0x800000000000808a,
    0x8000000080008000, 0x000000000000808b, 0x0000000080000001,
    0x8000000080008081, 0x8000000000008009, 0x000000000000008a,
    0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
    0x000000008000808b, 0x800000000000008b, 0x8000000000008089,
    0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
    0x000000000000800a, 0x800000008000000a, 0x8000000080008081,
    0x8000000000008080, 0x0000000080000001, 0x8000000080008008
], dtype=np.uint64)

def ROTL64(x, y):
    return ((x << y) & 0xFFFFFFFFFFFFFFFF) | (x >> (64 - y))

ROTATION_OFFSETS = np.array([
    [0, 36, 3, 41, 18],
    [1, 44, 10, 45, 2],
    [62, 6, 43, 15, 61],
    [28, 55, 25, 21, 56],
    [27, 20, 39, 8, 14]
], dtype=np.int64)


@jit(nopython=True)
def keccak_f(A, keccak_states):
    for round_index in range(NUM_ROUNDS):
        A = theta(A)
        keccak_states[round_index + 1, 0, :, :] = A
        A = rho(A)
        keccak_states[round_index + 1, 1, :, :] = A
        A = pi(A)
        keccak_states[round_index + 1, 2, :, :] = A
        A = chi(A)
        keccak_states[round_index + 1, 3, :, :] = A
        A = iota(A, round_index)
        keccak_states[round_index + 1, 4, :, :] = A
    return A


@jit(nopython=True)
def theta(A):
    C = np.zeros(5, dtype=np.uint64)
    D = np.zeros(5, dtype=np.uint64)
    
    # Accumulate XOR across columns for each of the 5 rows
    for x in range(5):
        C[x] = A[x, 0] ^ A[x, 1] ^ A[x, 2] ^ A[x, 3] ^ A[x, 4]
    
    # Calculate D array based on C
    for x in range(5):
        D[x] = C[(x-1) % 5] ^ C[(x+1) % 5]
    
    # Apply the D array to the A state
    for x in range(5):
        for y in range(5):
            A[x, y] ^= D[x]
    
    return A

@jit(nopython=True)
def rho(A):
    for x in range(5):
        for y in range(5):
            rotation_amount = ROTATION_OFFSETS[x,y]
            value = A[x, y]
            # Perform the rotation with wrap-around explicitly handled
            rotated_value = ((value << rotation_amount) | (value >> (64 - rotation_amount))) & ((1 << 64) - 1)
            A[x, y] = rotated_value
    return A

@jit(nopython=True)
def pi(A):
    B = np.zeros_like(A)
    for x in range(5):
        for y in range(5):
            B[y, (2*x + 3*y) % 5] = A[x, y]
    return B

@jit(nopython=True)
def chi(A):
    B = np.zeros_like(A)
    for x in range(5):
        for y in range(5):
            B[x, y] = A[x, y] ^ ((~A[(x+1) % 5, y]) & A[(x+2) % 5, y])
    return B

@jit(nopython=True)
def iota(A, round_index):
    A[0][0] ^= ROUND_CONSTANTS[round_index]
    return A