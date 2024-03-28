from numba import jit
import numpy as np
from sha3.absorb import absorb
from sha3.squeeze import squeeze

# @jit(nopython=True)
def sha3_256(input_bytes):
    # SHA3-256 specific parameters
    bit_rate = 1088
    output_length_bits = 256  # output length in bits
    
    # Initialize the state as a 5x5 matrix of 64-bit words
    state = np.zeros((5, 5), dtype=np.uint64)

    # Preprocess and pad the input bytes
    preprocessed_bytes = preprocess_and_pad(input_bytes, bit_rate // 8)

    print(bytes(preprocessed_bytes).hex())
    # return

    keccak_states = np.zeros((25, 5, 5, 5), dtype=np.uint64)

    # Absorb phase
    state = absorb(preprocessed_bytes, bit_rate, state, keccak_states)

    for i in range(24):
        for n in range(5):
            print(f'Keccak State Round {i}, Function {n}:')
            for y in range(5):
                for x in range(5):
                    print(f"[{x}, {y}] = {keccak_states[i, n, x, y]:016x}")
    
    # Squeeze phase - output length is passed in bits
    output_array = squeeze(state, output_length_bits)

    return output_array.tobytes()

# def preprocess_and_pad(input_bytes, byte_rate):
#     """
#     Apply SHA-3 pad10*1 padding and ensure each segment is correctly sized.

#     Parameters:
#     - input_bytes: The original input bytes to be hashed.
#     - byte_rate: The rate (in bytes) at which the input bytes are processed.
    
#     Returns:
#     - A bytes object that has been padded according to SHA-3 requirements
#       and is segmented correctly for processing.
#     """

#     input_bytes = input_bytes + b'\x06'  # Adding the delimiter for SHA-3

#     # Apply the SHA-3 pad10*1 padding
#     padding_len = byte_rate - (len(input_bytes) % byte_rate)
#     if padding_len == 1:
#         padded_input = input_bytes + b'\x81'  # Directly apply the 0x80 padding if only one byte needed
#     else:
#         padded_input = input_bytes + b'\x01' + (b'\x00' * (padding_len - 2)) + b'\x80'
    
#     return padded_input

def preprocess_and_pad(input_bytes, byte_rate):
    """
    Apply SHA-3 pad10*1 padding and ensure each segment is correctly sized.

    Parameters:
    - input_bytes: The original input bytes to be hashed.
    - byte_rate: The rate (in bytes) at which the input bytes are processed.
    
    Returns:
    - A bytes object that has been padded according to SHA-3 requirements
      and is segmented correctly for processing.
    """
    
    # Append the SHA-3 domain separation suffix '06' to the input
    # input_bytes += b'\x06'
    
    # Calculate the number of padding bytes required to make the total length a multiple of the byte rate
    padding_len = byte_rate - (len(input_bytes) % byte_rate)
    
    # Apply the SHA-3 pad10*1 padding scheme:
    # If exactly one byte of padding is needed, we use '\x86' to include both the '01' and the final '1' bits
    if padding_len == 1:
        padded_input = input_bytes + b'\x86'
    else:
        # In all other cases, we add '\x01', followed by necessary '\x00' padding bytes,
        # and end with '\x80' to include the final '1' bit
        padded_input = input_bytes + b'\x06' + (b'\x00' * (padding_len - 2)) + b'\x80'
    
    return padded_input
