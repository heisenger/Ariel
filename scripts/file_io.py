import numpy as np


def read_files(target_star, gt_df):
    """
    input: target_star ID and ground truth
    returns: airs_df, fgs_df, gt (adjusted align wavelength ) for the specific star
    output shape: observation * t * freq * spatial

    """
    airs_df = np.load(f"../data/processed/{target_star}_airs.npy")
    fgs_df = np.load(f"../data/processed/{target_star}_fgs1.npy")
    gt = gt_df[gt_df.planet_id == target_star].values[0][::-1]
    return airs_df, fgs_df, gt
