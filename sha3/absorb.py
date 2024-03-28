import numpy as np
from numba import jit, uint64
from sha3.keccak_f import keccak_f

@jit(nopython=True)
def bytes_to_int(segment):
    """
    Manually implement the conversion of a bytes object to an unsigned integer.
    This mirrors the functionality of int.from_bytes(segment, 'little').
    """
    result = uint64(0)
    for i in range(len(segment)):
        result |= uint64(segment[i]) << (i * 8)
    return result

@jit(nopython=True)
def absorb(preprocessed_input, bit_rate, state, keccak_states):
    """
    Absorb the preprocessed input bytes into the state with the given bitrate.
    """
    byte_rate = bit_rate // 8  # Convert bit rate to byte rate for processing
    num_blocks = len(preprocessed_input) // byte_rate

    for i in range(num_blocks):
        block = preprocessed_input[i * byte_rate: (i + 1) * byte_rate]
        
        for y in range(5):
            for x in range(5):
                if 8 * (x + 5 * y) < bit_rate: # TODO: remove this check (unnecessary with the preprocessing)
                    segment = block[8 * (x + 5 * y) // 8: 8 * (x + 5 * y) // 8 + 8]
                    if len(segment) == 8:
                        segment_int = bytes_to_int(segment)
                        state[x, y] ^= segment_int


        keccak_states[0, i, :, :] = state
        # for x in range(5):
        #     print(bytes_to_int(state[x]))

        state = keccak_f(state, keccak_states)

    return state