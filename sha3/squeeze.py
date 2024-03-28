from numba import jit
import numpy as np
from sha3.keccak_f import keccak_f

# @jit(nopython=True)
# def squeeze(state, output_length):
#     # Initialize an empty list to collect output bits
#     output_bits = []
    
#     # Convert the desired output length from bits to bytes
#     output_length_bytes = output_length // 8
    
#     # Iterate over the state to extract the output bytes
#     while len(output_bits) < output_length:
#         for x in range(5):
#             for y in range(5):
#                 # Convert the 64-bit word to bytes and add it to the output bits list
#                 for bit_pos in range(64):
#                     if len(output_bits) < output_length:
#                         output_bits.append((state[x, y] >> bit_pos) & 1)
#                     else:
#                         break
#                 if len(output_bits) >= output_length:
#                     break
#             if len(output_bits) >= output_length:
#                 break
        
#         # If more output is needed, permute the state again
#         if len(output_bits) < output_length:
#             state = keccak_f(state)

#     # Convert the output bits to bytes
#     output_bytes = bytearray()
#     for i in range(0, len(output_bits), 8):
#         byte = 0
#         for bit in range(8):
#             if i + bit < len(output_bits):
#                 byte |= (output_bits[i + bit] << bit)
#         output_bytes.append(byte)

#     return bytes(output_bytes[:output_length_bytes])

@jit(nopython=True)
def squeeze(state, output_length):
    # Calculate the number of output bytes
    output_length_bytes = output_length // 8
    
    # Initialize an empty array for output bytes
    output_bytes = np.zeros(output_length_bytes, dtype=np.uint8)
    
    # Track the current bit position in the output
    bit_idx = 0
    
    # Iterate over the state to extract the output bytes
    for x in range(5):
        for y in range(5):
            if bit_idx // 8 >= output_length_bytes:
                break  # Stop if we have enough output bytes
            word = state[x, y]
            for bit_pos in range(64):
                if bit_idx // 8 >= output_length_bytes:
                    break  # Stop if we have enough output bytes
                # Extract the bit at bit_pos
                bit = (word >> bit_pos) & 1
                # Place the bit in the correct position in the output_bytes array
                byte_idx = bit_idx // 8
                bit_in_byte_idx = bit_idx % 8
                output_bytes[byte_idx] |= np.uint8(bit << bit_in_byte_idx)
                bit_idx += 1

            if bit_idx // 8 >= output_length_bytes:
                break  # Stop if we have enough output bytes

    return output_bytes
