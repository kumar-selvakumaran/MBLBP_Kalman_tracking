from tqdm import tqdm
import numpy as np
import os
import sys

from MBLBP import multi_block_batch, lbp_batch, get_search_dist_mat

def test_multi_block_batch(sample_input = None):
    print(f"\n testing multi_block_batch()...")
    input_given = False
    if sample_input is None:
        a = np.random.rand(100,40,3,9,9)*255
        k = 40

    else:
        a = sample_input
        _, k, _, _, _ = sample_input.shape
        input_given = True

    windows = multi_block_batch(a, k)

    num_patches, num_points, num_channels, _, _ = a.shape

    for patch_ind in tqdm(range(num_patches)):
        for point_ind in range(num_points):
            for channel_ind in range(num_channels):
                testblock = np.ones([3,3])
                for i in range(3):
                    for j in range(3):
                        testblock[i,j] = a[patch_ind, point_ind, channel_ind, i*3:i*3+3, j*3:j*3+3].sum()

                if not np.array_equal((testblock *1e8).astype(int), (windows[patch_ind, point_ind,channel_ind]*1e8).astype(int)):
                    print(f"\n\nPRECISION ERROR : {(testblock *1e8).astype(int), (windows[patch_ind, point_ind,channel_ind]*1e8).astype(int)}")
                    break


def test_lbp_batch(sample_input = None):
    print(f"\n testing lbp_batch()...")
    input_given = False
    if sample_input is None:
        a = np.random.rand(100,40,3,9,9)*255
        k = 40

    else:
        a = sample_input
        _, k, _, _, _ = sample_input.shape
        input_given = True

    windows = multi_block_batch(a, k)

    lbps = lbp_batch(windows)

    num_patches, num_points, num_channels, _, _ = a.shape

    for patch_ind in tqdm(range(num_patches)):
        for point_ind in range(num_points):
            for channel_ind in range(num_channels):
                mask = np.ones([3,3], dtype=bool)
                mask[1, 1] = False

                test_window = windows[patch_ind, point_ind, channel_ind]
                binary_pattern = np.array(test_window[mask]>test_window[1,1]).astype(int)

                if not np.array_equal(binary_pattern, lbps[patch_ind, point_ind, channel_ind, :]):
                    print(f"\n MISMATCH ERROR : {binary_pattern, lbps[patch_ind, point_ind, channel_ind, :]}")
      
          

def test_get_search_dist_mat(sample_input = None):
    print(f"\n testing get_search_dist_mat()...")
    input_given = False
    if sample_input is None:
        a = np.random.rand(100,40,3,9,9)*255
        k = 40

    else:
        a = sample_input
        _, k, _, _, _ = sample_input.shape
        input_given = True

    windows = multi_block_batch(a, k)
    lbps = lbp_batch(windows)

    num_patches, num_points, num_channels, _, _ = a.shape
    rand_bs = np.random.randint(0, 2, size=(40, 3, 8))

    code_scores = get_search_dist_mat(lbps, rand_bs)

    for patch_ind in tqdm(range(num_patches)):
        for point_ind in range(num_points):
            for channel_ind in range(num_channels):

                binary_pattern = lbps[patch_ind, point_ind, channel_ind]

                xor_result = np.bitwise_xor(binary_pattern, rand_bs)
                test_scores = np.sum(xor_result.astype(int), axis=-1)

                if code_scores[patch_ind, point_ind, channel_ind] != test_scores[point_ind, channel_ind]:
                    print(f"\n MISMATCH ERROR : code score : {code_scores[patch_ind, point_ind, channel_ind]}, test score: {test_scores[point_ind, channel_ind]}")
      
def main():
    test_multi_block_batch()
    test_lbp_batch()
    test_get_search_dist_mat()

if __name__ == "__main__":
    main()


