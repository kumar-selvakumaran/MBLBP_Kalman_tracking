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



def get_search_space(img, patch_width, patch_height, stride_width, stride_height):
    """
    Extracts stride-wise windows from a 2D matrix using the sliding_window_view method.
    
    Parameters:
    - matrix (np.array): The input 2D matrix.
    - window_size (tuple): The height and width of the windows.
    - stride (tuple): The vertical and horizontal stride between windows.
    
    Returns:
    - np.array: A 4D array containing the stride-wise windows.
    """

    # Create the sliding windows
    search_patches =  np.lib.stride_tricks.sliding_window_view(img, (patch_width, patch_height), axis = (0,1))
    
    # Extract windows based on stride
    return search_patches[::stride_width, ::stride_height].reshape(-1, 3, patch_width, patch_height)


def init_feature_points(width, height, k):
    """
    Samples k random coordinates uniformly across the entire bounding box.

    Parameters:
    - width (int): The width of the bounding box.
    - height (int): The height of the bounding box.
    - k (int): The number of coordinates to sample.

    Returns:
    - np.array: An array of shape (k, 2) containing the sampled coordinates.
    """
    # Generate random x and y coordinates
    x_coords = np.random.uniform(0, width, size=k)
    y_coords = np.random.uniform(0, height, size=k)
    
    # Combine x and y coordinates
    coords = np.column_stack((x_coords, y_coords))

    coords = (coords - coords.min(axis = 0)) / (coords.max(axis = 0) - coords.min(axis = 0))
    
    return coords


def init_feature_points(width, height, k):
    all_coords = np.indices((height, width)).reshape(2, -1).T
  
    if k > len(all_coords):
      raise ValueError("samples required is greater than the pixels in input space : k > width*height")

    coords = all_coords[np.random.choice(len(all_coords), k ,replace=False)]
    print(coords.shape)

    coords = (coords - coords.min(axis = 0)) / (coords.max(axis = 0) - coords.min(axis = 0))

    return coords


def make_blocks_for_lbp(images, feature_points):
    """
    Extracts 9x9 windows centered on feature points from a set of BGR images.

    Parameters:
    - images (np.array): Array of images with shape [n, c, w, h].
    - feature_points (np.array): Array of feature points with shape [p, 2] (each row is [x, y]).

    Returns:
    - np.array: Extracted windows with shape [n, p, 3, 9, 9].
    """
    n, c, w, h = images.shape
    p = feature_points.shape[0]
    window_size = 9
    half_window = window_size // 2
    
    # Ensure points are within valid range
    # feature_points = np.clip(feature_points, half_window, min(w, h) - half_window - 1)
    
    # Generate relative indices for the window
    offset = np.arange(-half_window, half_window + 1)
    dx, dy = np.meshgrid(offset, offset, indexing='ij')
    
    # Combine all offsets with feature points for indexing
    all_x = feature_points[:, 0, np.newaxis, np.newaxis] + dx
    all_y = feature_points[:, 1, np.newaxis, np.newaxis] + dy
    
    # Extract windows using advanced indexing
    # Windows shape will be [p, n, c, 9, 9], need to transpose to [n, p, c, 9, 9]
    blocks = images[:, :, all_x, all_y].transpose((1, 0, 2, 3, 4))

    return blocks