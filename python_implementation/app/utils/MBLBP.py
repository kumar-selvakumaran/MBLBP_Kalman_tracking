import numpy as np
import os
import numpy as np

# TESTED
def multi_block_batch(blocks, k):
    """
    expected input as shape : [sxkx3x9x9] i.e. 'sxkx3x9x9'
    maybe 5-10 objects will be there just for loop it
    s : search space for a given target. #patches to search before determined lost
    k : number of randomly selected points for a given patch
    9x9x3 : size of a BGR block

    """
    blocks_reshaped = blocks.reshape(-1, k, 3, 3, 3, 3, 3)
    blocks_reshaped = blocks_reshaped.swapaxes(4, 5)
    blocks_reduced = blocks_reshaped.sum(axis=(5, 6))
    return blocks_reduced

# TESTED
def lbp_batch(windows):
    """
    expected input as shape : [sxkx3x3x3]
    maybe 5-10 objects will be there just for loop it
    s : search space for a given target. #patches to search before determined lost
    k : number of randomly selected points for a given patch
    for each window :

    Makes a local binary pattern for the given window in row major order like so
      1,2,3
      4, ,5
      6,7,8

    The peripheral values are thresholded using the center value of the given 3x3 window, where
    greater pixel are alloted '1' and '0' otherwise.
    '''
    """

    centers = windows[:, :, :, 1, 1]  # Extract the center of each window
    mask = np.ones_like(windows, dtype=bool)
    mask[:, :, :, 1, 1] = False

    # We use broadcasting to compare all values in the window except the center against the center value
    features = windows[mask].reshape(list(windows.shape[:-3]) + [3,8]) > centers[:, :, :, None]
    return features.astype(int)

# TESTED
# - check if a single object can have multiple patches.
# - may need to add up channel scores to have 1 score per pixel
def get_search_point_correspondence(lbps, target_patch):
    """
    expected input :
    1. lbps : local binary patterns of 's' patches (search space) [sxkx3x8]
    2. target_patch : k binary pattern feature vector of the target [kx3x8]
    maybe 5-10 objects will be there just for loop it
    s : search space for a given target. #patches to search before determined lost
    k : number of randomly selected points for a given patch
    for each window

    Given the an array of 'k' local binary patterns corresponding to each of
    the randomly selected points for the given
    """
    xor_result = np.bitwise_xor(lbps, target_patch)
    xor_difference_scores = np.sum(xor_result.astype(int), axis=-1)
    return xor_difference_scores