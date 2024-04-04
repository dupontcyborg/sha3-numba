"""SHA3 implementation in Python in OOP style"""

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

def rol(x, s):
    """Rotate x left by s."""
    return ((np.uint64(x) << np.uint64(s)) ^ (np.uint64(x) >> np.uint64(64 - s)))

class Sha3(object):
    """
    Implementation of the SHA-3 hash function.

    Args:
        bit_length (int): The desired bit length of the hash output.
        b (bytes): The input data to be hashed.

    Raises:
        ValueError: If the `bit_length` is not a valid value.

    Attributes:
        bit_length (int): The bit length of the hash output.
        rate (int): The rate of the sponge construction.
        dsbyte (int): The domain separation byte.
        i (int): The index of the current position in the buffer.
        state (numpy.ndarray): The internal state of the hash function.
        buf (numpy.ndarray): The buffer used for absorbing and squeezing.
        direction (int): The direction of the sponge construction.

    Methods:
        update(b): Absorb additional input data.
        digest(): Return the hash value as bytes.
        hexdigest(): Return the hash value as a hexadecimal string.

    """

    def __init__(self, bit_length, b=''):
        rate_map = {224: 144, 256: 136, 384: 104, 512: 72}
        if bit_length not in rate_map:
            raise ValueError("Invalid bit length.")

        self.bit_length = bit_length
        rate = rate_map[bit_length]
        self.rate, self.dsbyte, self.i = rate, _DSBYTE, 0
        self.state = np.zeros(25, dtype=np.uint64)
        self.buf = np.zeros(200, dtype=np.uint8)
        self._absorb(b)
        self.direction = _SPONGE_ABSORBING

    def update(self, b):
        """
        Absorb additional input data.

        Args:
            b (bytes): The input data to be absorbed.

        Returns:
            Sha3: The updated Sha3 object.

        """
        if self.direction == _SPONGE_SQUEEZING:
            self._permute()
        self._absorb(b)
        return self

    def digest(self):
        """
        Return the hash value as bytes.

        Returns:
            bytes: The hash value.

        """
        if self.direction == _SPONGE_ABSORBING:
            self._pad()
        return self._squeeze((200 - self.rate) // 2)

    def hexdigest(self):
        """
        Return the hash value as a hexadecimal string.

        Returns:
            str: The hash value as a hexadecimal string.

        """
        return self.digest().hex()

    def _keccak_f(self):
        """
        The keccak_f permutation function.

        """
        state = self.state
        bc = np.zeros(5, dtype=np.uint64)

        for i in range(24):
            # Parity
            for x in range(5):
                bc[x] = 0
                for y in range(0, 25, 5):
                    bc[x] ^= state[x + y]

            # Theta
            for x in range(5):
                t = bc[(x + 4) % 5] ^ rol(bc[(x + 1) % 5], 1)
                for y in range(0, 25, 5):
                    state[y + x] ^= t

            # Rho and pi
            t = state[1]
            for x in range(24):
                bc[0] = state[_KECCAK_PI[x]]
                state[_KECCAK_PI[x]] = rol(t, _KECCAK_RHO[x])
                t = bc[0]

            for y in range(0, 25, 5):
                for x in range(5):
                    bc[x] = state[y + x]
                for x in range(5):
                    state[y + x] = bc[x] ^ ((~bc[(x + 1) % 5]) & bc[(x + 2) % 5])

            state[0] ^= _KECCAK_RC[i]
        self.state = state

    def _absorb(self, b):
        """
        Absorb input data into the sponge construction.

        Args:
            b (bytes): The input data to be absorbed.

        """
        todo = len(b)
        i = 0
        while todo > 0:
            cando = self.rate - self.i
            willabsorb = min(cando, todo)
            self.buf[self.i:self.i + willabsorb] ^= \
                np.frombuffer(b[i:i+willabsorb], dtype=np.uint8)
            self.i += willabsorb
            if self.i == self.rate:
                self._permute()
            todo -= willabsorb
            i += willabsorb

    def _squeeze(self, n):
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
            cansqueeze = self.rate - self.i
            willsqueeze = min(cansqueeze, tosqueeze)
            b += self.state.view(dtype=np.uint8)[self.i:self.i + willsqueeze].tostring()
            self.i += willsqueeze
            if self.i == self.rate:
                self._permute()
            tosqueeze -= willsqueeze
        return b

    def _pad(self):
        """
        Pad the input data in the buffer.

        """
        self.buf[self.i] ^= self.dsbyte
        self.buf[self.rate - 1] ^= 0x80
        self._permute()

    def _permute(self):
        """
        Permute the internal state and buffer.

        """
        self.state ^= self.buf.view(dtype=np.uint64)
        self._keccak_f()
        self.i = 0
        self.buf[:] = 0

    def __repr__(self):
        return f'Sha3(bits={self.bit_length}, rate={self.rate}, dsbyte=0x{self.dsbyte:02x})'