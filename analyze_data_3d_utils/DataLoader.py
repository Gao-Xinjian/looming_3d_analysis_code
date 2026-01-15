import numpy as np
import pandas as pd
from scipy.io import loadmat
import os


def load_coord_data(mat_dir, pred_type='pred', check_sess=True):
    """
    Load coordinate data from .mat files
    
    Args:
        mat_dir: directory containing .mat files
        pred_type: 'pred' or 'com'
        check_sess: whether to verify session consistency
    
    Returns:
        dict with session IDs as keys and coordinate arrays as values
    """
    coord_dict = {}
    
    mat_files = [f for f in os.listdir(mat_dir) if f.endswith('.mat')]
    
    for mat_file in mat_files:
        try:
            mat_path = os.path.join(mat_dir, mat_file)
            mat_data = loadmat(mat_path)
            
            # Extract session ID from filename
            # Assuming format like "G1M1D1_pred.mat"
            sess_id = mat_file.split('_')[0]
            
            # Load prediction or COM data
            if pred_type == 'pred':
                key = 'pred' if 'pred' in mat_data else 'predictions'
            else:  # 'com'
                key = 'com' if 'com' in mat_data else 'center_of_mass'
            
            if key in mat_data:
                coord_dict[sess_id] = mat_data[key]
        except Exception as e:
            print(f"Error loading {mat_file}: {e}")
    
    return coord_dict


def load_syl_data(syl_dir, prefix, suffix, check_sess=True):
    """
    Load syllable/state data from CSV files
    
    Args:
        syl_dir: directory containing syllable CSV files
        prefix: filename prefix to filter
        suffix: filename suffix to filter
        check_sess: whether to verify session consistency
    
    Returns:
        dict with session IDs as keys and syllable arrays as values
    """
    syl_dict = {}
    
    csv_files = [f for f in os.listdir(syl_dir) if f.endswith('.csv')]
    
    for csv_file in csv_files:
        if prefix in csv_file and suffix in csv_file:
            try:
                csv_path = os.path.join(syl_dir, csv_file)
                df = pd.read_csv(csv_path)
                
                # Extract session ID
                # Assuming format like "-home-gxj-Desktop-Synology25-dannce_results-G1M1D1-save_data_AVG0.csv"
                parts = csv_file.split('-')
                for i, part in enumerate(parts):
                    if part.startswith('G') and 'M' in part and 'D' in part:
                        sess_id = part
                        break
                
                # Extract syllable column (usually first data column)
                if df.shape[1] > 0:
                    syl_dict[sess_id] = df.iloc[:, 0].values.astype(int)
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
    
    return syl_dict


def load_lsf_csv(csv_path, check_sess=True):
    """
    Load looming start frame (lsf) from CSV file
    
    Args:
        csv_path: path to CSV file containing lsf data
        check_sess: whether to verify session consistency
    
    Returns:
        dict with session IDs as keys and lists of lsf values
    """
    lsf_dict = {}
    
    try:
        df = pd.read_csv(csv_path)
        
        # Assuming columns like: sess_id, trial_id, lsf_start, ...
        for _, row in df.iterrows():
            sess_id = row.get('sess_id') or row.iloc[0]
            lsf_val = row.get('lsf_start') or row.get('lsf') or row.iloc[-1]
            
            if sess_id not in lsf_dict:
                lsf_dict[sess_id] = []
            
            lsf_dict[sess_id].append(int(lsf_val))
    except Exception as e:
        print(f"Error loading lsf CSV: {e}")
    
    return lsf_dict
