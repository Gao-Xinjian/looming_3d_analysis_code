import numpy as np
import pandas as pd


def calculate_speed(coords, unit='cm/s', joint_idx=0, exclude_z=True):
    """
    Calculate velocity and speed from coordinates
    
    Args:
        coords: array of shape (n_frames, n_joints, 3)
        unit: output unit (cm/s, mm/s, etc.)
        joint_idx: which joint to calculate for
        exclude_z: whether to exclude z coordinate
    
    Returns:
        velocity array and speed array
    """
    joint_coords = coords[:, joint_idx, :2] if exclude_z else coords[:, joint_idx, :3]
    
    # Compute displacement
    displacement = np.diff(joint_coords, axis=0, prepend=joint_coords[[0]])
    
    # Compute velocity magnitude
    speed = np.linalg.norm(displacement, axis=1)
    
    # Unit conversion if needed
    if unit == 'cm/s':
        speed = speed * 20  # Assuming 20 fps
    
    return displacement, speed


def calculate_distance(coords, joint_idx=0, target_point=None, exclude_z=True):
    """
    Calculate distance to target point
    
    Args:
        coords: array of shape (n_frames, n_joints, 3)
        joint_idx: which joint to calculate for
        target_point: target coordinates [x, y] or [x, y, z]
        exclude_z: whether to exclude z coordinate
    
    Returns:
        distances and relative vectors
    """
    if target_point is None:
        target_point = [0, 0]
    
    joint_coords = coords[:, joint_idx, :2] if exclude_z else coords[:, joint_idx, :3]
    target_point = np.array(target_point)
    
    # Compute vectors and distances
    vectors = joint_coords - target_point
    distances = np.linalg.norm(vectors, axis=1)
    
    return vectors, distances


def filter_behavior_in_time_range(bhvr_tuples_dict, time_range):
    """
    Filter behavior tuples within a time range
    
    Args:
        bhvr_tuples_dict: dict of behavior tuples
        time_range: (start, end) tuple
    
    Returns:
        filtered dict
    """
    filtered = {}
    start, end = time_range
    
    for sess_id, bhvr_dict in bhvr_tuples_dict.items():
        filtered[sess_id] = {}
        
        for bhvr_name, tuples in bhvr_dict.items():
            filtered_tuples = []
            
            for t_start, t_end in tuples:
                # Check if tuple overlaps with time range
                if t_start < end and t_end > start:
                    # Clip to time range
                    clipped_start = max(t_start, start)
                    clipped_end = min(t_end, end)
                    filtered_tuples.append((clipped_start, clipped_end))
            
            filtered[sess_id][bhvr_name] = filtered_tuples
    
    return filtered


def sync_start_time(bhvr_tuples_dict, sync_range):
    """
    Synchronize start time of behavior tuples
    
    Args:
        bhvr_tuples_dict: dict of behavior tuples
        sync_range: (new_start, end) tuple to synchronize to
    
    Returns:
        synchronized dict
    """
    synced = {}
    new_start, new_end = sync_range
    
    for sess_id, bhvr_dict in bhvr_tuples_dict.items():
        synced[sess_id] = {}
        
        for bhvr_name, tuples in bhvr_dict.items():
            synced_tuples = []
            
            for t_start, t_end in tuples:
                # Adjust time offset
                synced_start = t_start - new_start
                synced_end = t_end - new_start
                
                # Only include if within new range
                if synced_end > 0 and synced_start < (new_end - new_start):
                    synced_start = max(0, synced_start)
                    synced_end = min(new_end - new_start, synced_end)
                    synced_tuples.append((synced_start, synced_end))
            
            synced[sess_id][bhvr_name] = synced_tuples
    
    return synced
