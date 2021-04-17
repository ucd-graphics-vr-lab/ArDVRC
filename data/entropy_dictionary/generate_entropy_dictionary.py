import cv2
import cv2.aruco as aruco
import numpy as np
import math
import sys

np.set_printoptions(threshold=sys.maxsize)

def get_num_corners(dictionary):
    scores = []

    # for each marker in our dictionary, score it with the number of corners, which will be used to determine entropy
    for i in range(dictionary.bytesList.shape[0]):
        corners = get_corner_types_from_bit_mat(i, dictionary)
        scores.append((corners>0).sum())

    scores = np.array(scores)
    return scores

def get_dictionary_sorting_indices_by_entropy(dictionary):
    scores = get_num_corners(dictionary)

    # Sort in reverse order
    sorted_ids = np.argsort(scores)[::-1]

    return sorted_ids

def get_external_hamming_distances(dictionary):
    s = dictionary.bytesList.shape[0]
    hammings = []#np.zeros((s, s), dtype=np.int32)
    for i in range(dictionary.bytesList.shape[0]):
        h = []
        for j in range(dictionary.bytesList.shape[0]):
            if i == j:
                continue
            bit_mat_i = get_bit_mat(i, dictionary)
            bit_mat_j = get_bit_mat(j, dictionary)
            for k in range(4):
                bit_mat_j = np.rot90(bit_mat_j)
                xor_sum = np.bitwise_xor(bit_mat_i, bit_mat_j).sum()
                h.append(xor_sum)
        hammings.append(min(h))
    return hammings

def get_internal_hamming_distances(dictionary):
    hammings = []
    for i in range(dictionary.bytesList.shape[0]):
        h = []
        bit_mat1 = get_bit_mat(i, dictionary)
        bit_mat2 = bit_mat1.copy()
        for j in range(3):
            # rotate three times
            bit_mat2 = np.rot90(bit_mat2)
            #print('bit_mat2', bit_mat2.astype(np.int32))
            xor_sum = np.bitwise_xor(bit_mat1, bit_mat2).sum()
            h.append(xor_sum)
        hammings.append(min(h))
    return hammings

    #print('hammings min', min(hammings), 'hammings max', max(hammings))


def get_dictionary_sorting_indices_by_hamming(dictionary):
    scores = []

    # for each marker in our dictionary, score it with the number of corners, which will be used to determine entropy
    for i in range(dictionary.bytesList.shape[0]):
        corners = get_corner_types_from_bit_mat(i, dictionary)
        scores.append((corners>0).sum())

    scores = np.array(scores)

    # Sort in reverse order
    sorted_ids = np.argsort(scores)[::-1]

    return sorted_ids

def get_corner_types_from_bit_mat(id_num, dictionary):
    bit_mat = get_bit_mat(id_num, dictionary)

    m = dictionary.markerSize
    # Layout of corners should be of size markerSize + 2 (for edges) + 1 (one more corner than cells)
    s = m + 2 + 1
    code_mat = np.zeros((s,s), dtype=np.uint8)
    for i in range(s):
        for j in range(s):
            # Get four bits in a square to codify the corner type
            bits = np.hstack((np.zeros(4, dtype=np.bool), bit_mat[i:i+2, j:j+2].reshape(-1)))
            code_mat[j][i] = np.packbits(bits)
    # The edges/corners are now encoded as integers, but they contain orientation information.
    # I'm not sure we have  much use for this information, so I will simplify things and label
    # the information with 6 values instead of 16

    label_mat = np.zeros_like(code_mat)
    # Completely black = 0
    #label_mat[code_mat == 0] = 0
    # Completely white = 15
    #label_mat[code_mat == 15] = 0

    # Mosty white corner type = 1
    label_mat[code_mat == 7] = 1
    label_mat[code_mat == 11] = 1
    label_mat[code_mat == 13] = 1
    label_mat[code_mat == 14] = 1

    # Mostly black corner type = 2
    label_mat[code_mat == 1] = 2
    label_mat[code_mat == 2] = 2
    label_mat[code_mat == 4] = 2
    label_mat[code_mat == 8] = 2

    # Checkerboard corner type = 3
    label_mat[code_mat == 6] = 3
    label_mat[code_mat == 9] = 3
    # Simple edges are type 4
    #label_mat[code_mat == 3] = 4
    #label_mat[code_mat == 5] = 4
    #label_mat[code_mat == 10] = 4
    #label_mat[code_mat == 12] = 4
    return label_mat

def get_bit_mat(id_num, dictionary, rotation=0):
    """ Returns the bits for the marker, including outer white area and black area."""
    # TODO: Rotation not actually implemented in this instance
    assert(rotation >= 0)
    assert(rotation <= 3)

    bytes_list = dictionary.bytesList[id_num]

    # Make a 1D list of bytes
    bytes_list = bytes_list.reshape(-1)

    m = dictionary.markerSize

    # Determine how many bytes are needed to represent the data
    n_bytes = math.ceil((m*m)/8)

    # bit_mat shape should be the marker size + 2 (black solid border on each side) + 2 (solid white outer border)
    bit_mat = np.zeros((m + 4, m + 4), dtype=np.bool)

    # Fill in outer edges with ones
    bit_mat[0:,0] = 1
    bit_mat[0:,-1] = 1
    bit_mat[0,0:] = 1
    bit_mat[-1,0:] = 1

    # Determine how many bits we need
    total_bits = m*m

    # Our first full loop runs through all the bytes where all of the bits are used
    n = total_bits//8
    bits = None
    for i in range(n):
        b = np.unpackbits(np.uint8(bytes_list[rotation*n_bytes+i]))
        if bits is None:
            bits = b
        else:
            bits = np.concatenate((bits, b))

    # If we haven't covered all the bits, get the partial byte needed
    if n*8 != total_bits:
        b = np.unpackbits(np.uint8(bytes_list[rotation*n_bytes+i+1]))
        b = b[8-(total_bits%8):]
        bits = np.concatenate((bits, b))

    # Write the data in the center of the marker
    for i in range(m):
        bit_mat[i+2,2:2+m] = bits[m*i:m*(i+1)]

    for i in range(rotation):
        bit_mat = np.rot90(bit_mat)

    return bit_mat

# We will derive our custom entropy-ordered dictionary from the 4x4_1000 dictionary
# TODO: Expand to other dictionaries later?
dictionary = aruco.Dictionary_get(aruco.DICT_4X4_100)

#sorted_ids = [get_dictionary_sorting_indices_by_entropy(d) for d in dictionaries]
corners = get_num_corners(dictionary)

hamming_int = get_internal_hamming_distances(dictionary)

# Thresholding values, chosen through experimentation
HAMMING_THRESH = 6
CORNERS_THRESH = 17

hamming_int = np.array(hamming_int)

# Threshold markers based on hamming distance (internal), # of corners
hamming_int_thresh = hamming_int >= HAMMING_THRESH
corner_threshold = corners >= CORNERS_THRESH

# Get markers in common that have a high internal hamming and high number of corners
mask = np.bitwise_and(hamming_int_thresh, corner_threshold)

print('Number of IDs:', mask.sum())

id_vals = np.argwhere(mask>0)
n_corners = corners[id_vals]
n_corners = n_corners.reshape(-1)

# Save new dictionary as a NumPy array
new_dictionary = aruco.Dictionary_get(aruco.DICT_4X4_1000)
new_dictionary.bytesList = new_dictionary.bytesList[id_vals]
s = new_dictionary.bytesList.shape
if (len(s)==4):
    new_dictionary.bytesList = new_dictionary.bytesList.reshape((s[0], s[2], s[3]))

np.save('DICT_4X4_'+str(new_dictionary.bytesList.shape[0])+'_ENTROPY.npy', new_dictionary.bytesList)