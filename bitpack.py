import torch

def unpack32(values, bit_depths, repeats=4):
    # D = [7,1,3,3,1,7,2,3,5,1,2,4,1,2,7,1,1,8,6,1,4,2,1,4,3,4,4,1,3,2,1,1]
    # Validate input
    total_bits = 32 * values.shape[0]
    bit_depths = bit_depths.repeat_interleave(repeats);
    assert bit_depths.sum(dtype=torch.int) == total_bits, "Sum of bit depths must equal 32N"

    # Flatten the int32 array into a single bitstream
    bitstream = 0
    for i, value in enumerate(values):
        bitstream |= (value << (32 * i))  # Shift each int into its place

    # Split the bitstream into parts based on bit depths
    parts = torch.zeros([len(bit_depths), values.shape[1]], device=values.device)
    current_bit = 0

    for i, bits in enumerate(bit_depths):
        mask = (1 << bits) - 1  # Create a mask for `bits` number of bits
        parts[i,:] = (bitstream >> current_bit) & mask  # Extract and shift
        # parts.append(part)
        current_bit += bits  # Move to the next segment

    return parts
