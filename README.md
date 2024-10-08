# SHA-3 Numba

This repository implements the SHA-3 (Keccak) cryptographic hash function in Python with Numba for performance optimization, including both CPU and CUDA (GPU) support. The implementations showcase **significant** performance improvements through Numba’s just-in-time (JIT) compilation.

**NOTE: This code is meant for demonstrative purposes only. As with any cryptographic code, use at your own risk.**

## Performance

The performance of SHA-3 implementations across Python, Numba-optimized, and native libraries (hashlib) shows substantial improvements when leveraging Numba’s just-in-time (JIT) compilation. 

The benchmarks below show that the pure Python implementation is `~3,038x` slower than native code. With Numba acceleration, this reduces to `~2.7x` slower than native, a `~1,120x` improvement.

**SHA3-256 Input Length: 1000 bytes, Iterations: 1000 (M1 Max, single-thread)**
```
Pure Python:
- Time: 6.3804 s
- Hashes per second: 156.7

Numba CPU:
- Time: 0.0057 s
- Hashes per second: 174,700.5

Native (hashlib):
- Time: 0.0021 s
- Hashes per second: 479,003.6
```

## Getting Started

### Prerequisites

To run this repository, you need Python 3.11 and the following Python packages:

- `numpy`
- `numba`
- `ipykernel` (for running Jupyter notebooks, if needed)

In addition, if you'd like to run SHA-3 on GPU, you will need:

- `cuda-toolkit`

### Setup

1. Clone the repo

```
git clone https://github.com/dupontcyborg/sha3-numba.git
```

2. Install the dependencies

Using `pipenv`
```
pipenv install --dev
```

Alternatively, using `pip`
```
pip install -r requirements.txt
```

### Usage

To use SHA-3 Numba in your Python code, you can import any of the three options below:

```
from sha3 import sha3_python # Imports pure Python implementation (slow)

from sha3 import sha3_numba  # Imports Numba accelerated CPU implementation (fast)

from sha3 import sha3_cuda   # Imports Numba accelerated GPU implementation (needs CUDA)
```

### Benchmarking

To benchmark the three SHA-3 implementations against native code (`hashlib`), run the `benchmark.ipynb` notebook. This notebook benchmarks SHA-3 implementations in Pure Python against Numba.

## License

This repository is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more details.