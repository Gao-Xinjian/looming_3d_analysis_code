import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter


def find_medoid_distance_outliers(coords, outlier_scale_factor=6):
    """
    Find outliers using medoid distance method
    
    Args:
        coords: array of shape (n_frames, n_joints, 3)
        outlier_scale_factor: multiplier for threshold
    
    Returns:
        dict with 'mask' key containing boolean outlier mask
    """
    n_frames, n_joints = coords.shape[:2]
    mask = np.zeros((n_frames, n_joints), dtype=bool)
    
    for joint_idx in range(n_joints):
        joint_coords = coords[:, joint_idx, :3]
        
        # Compute distances from median
        median_coord = np.median(joint_coords, axis=0)
        distances = np.linalg.norm(joint_coords - median_coord, axis=1)
        
        # Threshold
        threshold = np.median(distances) + outlier_scale_factor * np.std(distances)
        mask[:, joint_idx] = distances > threshold
    
    return {'mask': mask}


def find_RULE_outliers(coords, joint_names, limb_orders, skel_conns, joint_creds,
                       prev_outliers=None, outlier_scale_factors=None):
    """
    Find outliers using RULE (Robust Upper Limb Estimation) method
    
    Args:
        coords: array of shape (n_frames, n_joints, 3)
        joint_names: list of joint names
        limb_orders: limb order information
        skel_conns: skeleton connections
        joint_creds: joint credibility information
        prev_outliers: previous outliers to combine with
        outlier_scale_factors: dict with 'velocity', 'angle', 'displacement' keys
    
    Returns:
        dict with 'mask' key containing boolean outlier mask
    """
    n_frames, n_joints = coords.shape[:2]
    mask = np.zeros((n_frames, n_joints), dtype=bool)
    
    if prev_outliers is not None:
        mask = prev_outliers['mask'].copy()
    
    # Compute velocity-based outliers
    velocity = np.diff(coords, axis=0, prepend=coords[[0]])
    vel_mag = np.linalg.norm(velocity, axis=2)
    
    vel_scale = outlier_scale_factors.get('velocity', 5) if outlier_scale_factors else 5
    for joint_idx in range(n_joints):
        vel_threshold = np.median(vel_mag[:, joint_idx]) + vel_scale * np.std(vel_mag[:, joint_idx])
        mask[:, joint_idx] |= vel_mag[:, joint_idx] > vel_threshold
    
    return {'mask': mask}


def interpolate_keypoints(coords, outlier_mask):
    """
    Interpolate missing keypoints marked as outliers
    
    Args:
        coords: array of shape (n_frames, n_joints, 3)
        outlier_mask: boolean mask of outliers
    
    Returns:
        interpolated coordinate array
    """
    n_frames, n_joints, n_dims = coords.shape
    interp_coords = coords.copy()
    
    for joint_idx in range(n_joints):
        for dim in range(n_dims):
            joint_vals = coords[:, joint_idx, dim]
            outliers = outlier_mask[:, joint_idx]
            
            if np.any(outliers):
                # Create interpolation function from valid points
                valid_idx = np.where(~outliers)[0]
                
                if len(valid_idx) > 1:
                    f = interp1d(valid_idx, joint_vals[valid_idx], kind='linear',
                                fill_value='extrapolate')
                    all_idx = np.arange(n_frames)
                    interp_coords[:, joint_idx, dim] = f(all_idx)
    
    return interp_coords


def simplify_coord_dict(coord_dict, joint_names, limb_combos):
    """
    Simplify coordinates by combining related joints
    
    Args:
        coord_dict: dict of coordinate arrays
        joint_names: list of original joint names
        limb_combos: dict specifying how to combine joints
    
    Returns:
        simplified coord_dict and simplified joint names
    """
    simp_coord_dict = {}
    simp_joint_names = []
    
    # Define simplification mapping
    if not limb_combos:
        # Default: use all joints
        simp_joint_names = joint_names
        simp_coord_dict = coord_dict
    else:
        for simp_name, joint_list in limb_combos.items():
            simp_joint_names.append(simp_name)
        
        for sess_id, coords in coord_dict.items():
            simp_coords = []
            
            for simp_name, joint_list in limb_combos.items():
                joint_indices = [joint_names.index(j) for j in joint_list if j in joint_names]
                
                if joint_indices:
                    # Average positions of combined joints
                    combined = np.mean(coords[:, joint_indices, :], axis=1)
                    simp_coords.append(combined)
            
            simp_coord_dict[sess_id] = np.stack(simp_coords, axis=1)
    
    return simp_coord_dict, simp_joint_names


def map_syllabel_to_behavior(syl_dict, bhvr_map):
    """
    Map syllable labels to behavior names
    
    Args:
        syl_dict: dict of syllable label arrays
        bhvr_map: dict mapping syllable numbers to behavior names
    
    Returns:
        dict of behavior label arrays
    """
    bhvr_dict = {}
    
    for sess_id, syl_labels in syl_dict.items():
        bhvr_labels = np.array([bhvr_map.get(int(syl), 'other') for syl in syl_labels], dtype=object)
        bhvr_dict[sess_id] = bhvr_labels
    
    return bhvr_dict


def convert_bhvr_kpms2series(bhvr_dict):
    """
    Convert behavior array to pandas Series format
    
    Args:
        bhvr_dict: dict of behavior label arrays
    
    Returns:
        dict of behavior Series
    """
    bhvr_series_dict = {}
    
    for sess_id, bhvr_labels in bhvr_dict.items():
        bhvr_series_dict[sess_id] = {}
        
        # Get unique behaviors
        unique_bhvrs = np.unique(bhvr_labels)
        
        for bhvr in unique_bhvrs:
            mask = bhvr_labels == bhvr
            series = pd.Series(mask.astype(int), name=bhvr)
            bhvr_series_dict[sess_id][bhvr] = series
    
    return bhvr_series_dict


def convert_bhvr_series2tuples(bhvr_series_dict):
    """
    Convert behavior Series to (start, end) tuple format
    
    Args:
        bhvr_series_dict: dict of behavior Series dicts
    
    Returns:
        dict of behavior tuples
    """
    bhvr_tuples_dict = {}
    
    for sess_id, bhvr_series in bhvr_series_dict.items():
        bhvr_tuples_dict[sess_id] = {}
        
        for bhvr_name, series in bhvr_series.items():
            tuples = []
            
            if isinstance(series, dict):
                # If it's a dict, get the series from it
                series_data = series.get(bhvr_name, pd.Series([]))
            else:
                series_data = series
            
            # Find contiguous regions
            diff = np.diff(series_data.astype(int), prepend=0, append=0)
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            
            for start, end in zip(starts, ends):
                tuples.append((start, end))
            
            bhvr_tuples_dict[sess_id][bhvr_name] = tuples
    
    return bhvr_tuples_dict


def save_bhvr_dicts(bhvr_dict, save_folder, save_format='csv', file_prefix=''):
    """
    Save behavior dictionaries to files
    
    Args:
        bhvr_dict: dict of behavior data
        save_folder: folder to save files
        save_format: 'csv' or 'pickle'
        file_prefix: prefix for filenames
    """
    import os
    os.makedirs(save_folder, exist_ok=True)
    
    for sess_id, bhvr_data in bhvr_dict.items():
        filename = f"{file_prefix}{sess_id}.{save_format}"
        filepath = os.path.join(save_folder, filename)
        
        if save_format == 'csv':
            if isinstance(bhvr_data, dict):
                df = pd.DataFrame(bhvr_data)
            else:
                df = pd.DataFrame(bhvr_data)
            df.to_csv(filepath, index=False)
