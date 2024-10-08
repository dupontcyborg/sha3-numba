import numpy as np
import hashlib
import time
import sys

sys.path.append('..')
from sha3 import sha3_python, sha3_numba, sha3_cuda

def benchmark_and_verify(bit_length, input_bytes, num_iterations, use_cuda=False):

    # Python SHA3 Implementation
    start_time = time.perf_counter()
    for _ in range(num_iterations):
        python_hash = sha3_python(bit_length, input_bytes).tobytes().hex()
    python_time = time.perf_counter() - start_time

    # Python SHA3 Implementation (Numba)
    sha3_numba(bit_length, input_bytes) # Warm up
    start_time = time.perf_counter()
    for _ in range(num_iterations):
        numba_hash = sha3_numba(bit_length, input_bytes).tobytes().hex()
    numba_time = time.perf_counter() - start_time

    # Python SHA3 Implementation (CUDA)
    if use_cuda:
        # Warm up
        sha3_cuda(bit_length, input_bytes, 1)

        start_time = time.perf_counter()
        cuda_hashes = sha3_cuda(bit_length, input_bytes, num_iterations)
        cuda_time = time.perf_counter() - start_time

        # Verify that all cuda hashes are the same
        cuda_hash = cuda_hashes[0]
        for i in range(1, len(cuda_hashes)):
            assert np.array_equal(cuda_hash, cuda_hashes[i]), f"Mismatch: {cuda_hash} != {cuda_hashes[i]}"
        cuda_hash = cuda_hash.tobytes().hex()

    # HashLib SHA3 Implementation
    hashlib_func = hashlib.sha3_256 if bit_length == 256 else hashlib.sha3_512
    start_time = time.perf_counter()
    for _ in range(num_iterations):
        lib_hash = hashlib_func(input_bytes).hexdigest()
    lib_time = time.perf_counter() - start_time
    
    print(f"Input Length: {len(input_bytes)} bytes, Iterations: {num_iterations}")
    print(f"Python  SHA3-{bit_length} Time: {python_time:.4f} s, Hashes/s: {num_iterations/python_time:.1f}, Length: {len(python_hash)}, Hash: {python_hash}")
    print(f"Numba   SHA3-{bit_length} Time: {numba_time:.4f} s, Hashes/s: {num_iterations/numba_time:.1f}, Length: {len(numba_hash)}, Hash: {numba_hash}")
    if use_cuda:
        print(f"CUDA    SHA3-{bit_length} Time: {cuda_time:.4f} s, Hashes/s: {num_iterations/cuda_time:.1f}, Length: {len(cuda_hash)}, Hash: {cuda_hash}")
    print(f"HashLib SHA3-{bit_length} Time: {lib_time:.4f} s, Hashes/s: {num_iterations/lib_time:.1f}, Length: {len(lib_hash)}, Hash: {lib_hash}")

    # Verify Outputs
    assert python_hash == lib_hash, f"Mismatch: {python_hash} != {lib_hash}"
    assert numba_hash == lib_hash, f"Mismatch: {numba_hash} != {lib_hash}"
    if use_cuda:
        assert cuda_hash == lib_hash, f"Mismatch: {cuda_hash} != {lib_hash}"
    print("Verification: PASSED\n")
