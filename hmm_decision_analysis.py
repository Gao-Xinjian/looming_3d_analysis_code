"""
HMM-based Decision Analysis for Mouse Looming Response

这个模块使用Hidden Markov Model从行为观测序列推断小鼠的决策状态。

核心概念:
- 观测 (Observations): 原始行为标签 + 运动学特征
- 隐藏状态 (Hidden States): 决策状态 (E, F, R, I)
- HMM自动学习: 观测→状态的概率映射 + 状态转移概率

Author: GXJ
Date: 2026-03-18
"""

import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import pickle


def extract_features_for_hmm(sess_id, bhvr_tuples_dict_sess, coord_dict_sess, 
                              nest_tuples_dict_sess, joint_names, fps=20):
    """
    Extract observation features for HMM from behavior and kinematic data.
    
    Features:
    - Raw behavior labels (walking, grooming, etc.)
    - Velocity (speed of movement)
    - Distance to nest
    - Acceleration
    
    Args:
        sess_id: session identifier
        bhvr_tuples_dict_sess: behavior intervals for this session
        coord_dict_sess: 3D coordinates (n_frames, n_joints, 3)
        nest_tuples_dict_sess: nest location intervals
        joint_names: list of joint names
        fps: frames per second
        
    Returns:
        features_dict: dict with frame indices as keys and feature vectors as values
    """
    # Get coordinates for this session
    coords = coord_dict_sess  # Shape: (n_frames, n_joints, 3)
    n_frames = coords.shape[0]
    
    # Get spine/com position (use joint index for spine or COM)
    spine_idx = joint_names.index('SpineM') if 'SpineM' in joint_names else 0
    positions = coords[:, spine_idx, :]  # (n_frames, 3)
    
    # Calculate velocity (mm/frame)
    velocity = np.zeros(n_frames)
    velocity[1:] = np.linalg.norm(positions[1:] - positions[:-1], axis=1)
    velocity = velocity * fps  # Convert to mm/s
    
    # Calculate acceleration
    acceleration = np.zeros(n_frames)
    acceleration[1:] = velocity[1:] - velocity[:-1]
    
    # Calculate distance to nest (assume nest is at specific location)
    # Method 1: Use nest_tuples_dict to estimate nest center from mouse positions when in nest
    nest_frames = []
    for nest_bhvr, intervals in nest_tuples_dict_sess.items():
        for start, end in intervals:
            # Ensure indices are within valid range
            valid_start = max(0, start)
            valid_end = min(end, n_frames - 1)
            if valid_start <= valid_end:
                nest_frames.extend(range(valid_start, valid_end + 1))
    
    if len(nest_frames) > 0:
        # Use median position when mouse is in nest as nest center
        nest_positions = positions[nest_frames]
        nest_center = np.median(nest_positions, axis=0)
    else:
        # Fallback: Use the 5th percentile of all positions (likely nest location)
        # Since mice tend to stay near nest, low percentile positions are likely nest area
        nest_center = np.percentile(positions, 5, axis=0)
        print(f"Warning: No nest frames found for {sess_id}, using 5th percentile position as nest center")
    
    dist_to_nest = np.linalg.norm(positions - nest_center, axis=1)
    
    # Create behavior sequence (label for each frame)
    behavior_sequence = ['unknown'] * n_frames
    
    for bhvr_name, intervals in bhvr_tuples_dict_sess.items():
        for start_frame, end_frame in intervals:
            for f in range(start_frame, min(end_frame+1, n_frames)):
                behavior_sequence[f] = bhvr_name
    
    # Combine features
    features_dict = {
        'behavior': behavior_sequence,
        'velocity': velocity,
        'acceleration': acceleration,
        'dist_to_nest': dist_to_nest,
        'n_frames': n_frames
    }
    
    return features_dict


def prepare_hmm_observation_sequences(sess_id, lsf_list, features_dict, 
                                       looming_duration=3.625, blank_time=2.0, 
                                       looming_rep=5, fps=20):
    """
    Prepare observation sequences for each looming stimulus.
    
    Args:
        sess_id: session identifier
        lsf_list: list of looming start frames
        features_dict: extracted features
        looming_duration: duration of looming stimulus (seconds)
        blank_time: blank period after looming (seconds)
        looming_rep: number of looming repetitions
        fps: frames per second
        
    Returns:
        looming_sequences: dict[trial_id][looming_id] = {'observations': array, 'length': int}
        behavior_encoder: LabelEncoder for behavior labels
    """
    looming_sequences = {}
    
    # Encode behaviors to numeric values
    behavior_encoder = LabelEncoder()
    all_behaviors = list(set(features_dict['behavior']))
    behavior_encoder.fit(all_behaviors)
    behavior_codes = behavior_encoder.transform(features_dict['behavior'])
    
    for t_idx, lsf in enumerate(lsf_list):
        trial_id = f'{sess_id}T{t_idx + 1}'
        looming_sequences[trial_id] = {}
        
        for looming_id in range(looming_rep):
            # Calculate time window for this looming
            lsf_start = int(lsf + looming_id * (looming_duration + blank_time) * fps)
            lsf_end = int(lsf_start + looming_duration * fps)
            looming_end = int(lsf_end + blank_time * fps)
            
            # Extract features for this time window
            if looming_end <= features_dict['n_frames']:
                # Create observation matrix: [behavior_code, velocity, acceleration, dist_to_nest]
                obs_behavior = behavior_codes[lsf_start:looming_end].reshape(-1, 1)
                obs_velocity = features_dict['velocity'][lsf_start:looming_end].reshape(-1, 1)
                obs_accel = features_dict['acceleration'][lsf_start:looming_end].reshape(-1, 1)
                obs_dist = features_dict['dist_to_nest'][lsf_start:looming_end].reshape(-1, 1)
                
                observations = np.hstack([obs_behavior, obs_velocity, obs_accel, obs_dist])
                
                looming_sequences[trial_id][looming_id] = {
                    'observations': observations,
                    'length': observations.shape[0],
                    'frame_range': (lsf_start, looming_end)
                }
    
    return looming_sequences, behavior_encoder


def train_hmm_model(all_observations, n_states=4, n_iter=100, random_state=42):
    """
    Train a Gaussian HMM model on all observation sequences.
    
    Args:
        all_observations: list of observation arrays
        n_states: number of hidden states (default 4: E, F, R, I)
        n_iter: number of training iterations
        random_state: random seed
        
    Returns:
        model: trained HMM model
    """
    # Concatenate all observations
    X = np.vstack(all_observations)
    lengths = [obs.shape[0] for obs in all_observations]
    
    # Train Gaussian HMM
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type='full',
        n_iter=n_iter,
        random_state=random_state,
        verbose=True
    )
    
    print(f"Training HMM with {n_states} states on {len(all_observations)} sequences...")
    model.fit(X, lengths)
    
    print(f"Training completed. Log-likelihood: {model.score(X, lengths):.2f}")
    print(f"\nTransition matrix:")
    print(model.transmat_)
    
    return model


def predict_decision_states(model, observations):
    """
    Predict hidden states (decision states) for given observations.
    
    Args:
        model: trained HMM model
        observations: observation array
        
    Returns:
        states: array of predicted state indices for each frame
    """
    states = model.predict(observations)
    return states


def interpret_hmm_states(states, state_mapping=None):
    """
    Interpret HMM states to decision labels (E, F, R, I).
    
    If state_mapping is None, automatically assign based on state statistics.
    
    Args:
        states: array of state indices
        state_mapping: dict mapping state index to decision label
        
    Returns:
        decision: most common state mapped to E/F/R/I
    """
    if len(states) == 0:
        return 'I'
    
    # Get most common state
    state_counts = Counter(states)
    most_common_state = state_counts.most_common(1)[0][0]
    
    if state_mapping is not None:
        return state_mapping.get(most_common_state, 'I')
    else:
        # Default: map state index to decision (needs manual interpretation)
        return f'S{most_common_state}'


def analyze_hmm_states(hmm_model, all_looming_sequences, n_decision_states):
    """
    Analyze HMM states to understand what each state represents.
    
    Args:
        hmm_model: trained HMM model
        all_looming_sequences: all observation sequences
        n_decision_states: number of hidden states
        
    Returns:
        state_features: dict with statistics for each state
    """
    print("Analyzing HMM states...\n")
    
    # Collect all states and their corresponding features
    state_features = defaultdict(lambda: {
        'velocity': [], 
        'acceleration': [], 
        'dist_to_nest': [], 
        'behaviors': []
    })
    
    for sess_id, trials in all_looming_sequences.items():
        for trial_id, loomings in trials.items():
            for looming_id, data in loomings.items():
                obs = data['observations']
                states = predict_decision_states(hmm_model, obs)
                
                for i, state in enumerate(states):
                    state_features[state]['velocity'].append(obs[i, 1])  # velocity column
                    state_features[state]['acceleration'].append(obs[i, 2])  # accel column
                    state_features[state]['dist_to_nest'].append(obs[i, 3])  # distance column
                    state_features[state]['behaviors'].append(obs[i, 0])  # behavior code
    
    # Print statistics for each state
    print("State characteristics (mean ± std):")
    print("="*80)
    for state in range(n_decision_states):
        print(f"\nState {state}:")
        print(f"  Count: {len(state_features[state]['velocity'])} frames")
        print(f"  Velocity: {np.mean(state_features[state]['velocity']):.2f} ± {np.std(state_features[state]['velocity']):.2f} mm/s")
        print(f"  Acceleration: {np.mean(state_features[state]['acceleration']):.2f} ± {np.std(state_features[state]['acceleration']):.2f}")
        print(f"  Distance to nest: {np.mean(state_features[state]['dist_to_nest']):.2f} ± {np.std(state_features[state]['dist_to_nest']):.2f} mm")
        
        # Most common behaviors in this state
        behavior_counts = Counter(state_features[state]['behaviors'])
        print(f"  Top behaviors: {behavior_counts.most_common(3)}")
    
    print("\n" + "="*80)
    print("Based on these characteristics, manually map states to decisions:")
    print("  - High velocity + high accel → likely Escape (E)")
    print("  - Low velocity + near nest → likely Freeze (F)")
    print("  - Medium velocity + toward nest → likely Retreat (R)")
    print("  - Low velocity + far from nest → likely Ignore (I)")
    
    return state_features


def calculate_trial_decision_hmm(looming_decisions):
    """
    Calculate trial-level decision from looming decisions (HMM version).
    Uses same logic as rule-based method for consistency.
    
    Args:
        looming_decisions: list of looming-level decisions
        
    Returns:
        str: trial-level decision
    """
    filtered = [d for d in looming_decisions if d != 'I']
    
    if len(filtered) == 0:
        return 'I'
    
    # Check for acute escape (first looming)
    first_decision = filtered[0]
    if first_decision == 'E':
        return 'A+E'
    
    # Collect all decision types
    has_E = 'E' in filtered
    has_F = 'F' in filtered
    has_R = 'R' in filtered
    
    # Combine
    if has_E and has_F:
        return 'F+E'
    elif has_E and has_R:
        return 'R+E'
    elif has_F and has_R:
        return 'F+R'
    elif has_E:
        return 'E'
    elif has_F:
        return 'F'
    elif has_R:
        return 'R'
    else:
        return 'I'


def compare_decisions(deci_trial_dict, hmm_trial_decisions):
    """
    Compare HMM decisions with rule-based decisions.
    
    Args:
        deci_trial_dict: rule-based trial decisions
        hmm_trial_decisions: HMM-based trial decisions
        
    Returns:
        df_comparison: DataFrame with comparison results
    """
    print("Comparing HMM vs Rule-based decisions...\n")
    
    # Compare trial-level decisions
    comparison_data = []
    
    for sess_id in deci_trial_dict.keys():
        if sess_id in hmm_trial_decisions:
            for trial_id in deci_trial_dict[sess_id].keys():
                if trial_id in hmm_trial_decisions[sess_id]:
                    rule_decision = deci_trial_dict[sess_id][trial_id]
                    hmm_decision = hmm_trial_decisions[sess_id][trial_id]
                    
                    comparison_data.append({
                        'trial_id': trial_id,
                        'rule_based': rule_decision,
                        'hmm_based': hmm_decision,
                        'match': rule_decision == hmm_decision
                    })
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Calculate agreement
    agreement = df_comparison['match'].sum() / len(df_comparison) * 100
    print(f"Agreement between methods: {agreement:.1f}%\n")
    
    # Show confusion matrix
    print("Confusion matrix (rows=Rule-based, cols=HMM):")
    confusion = pd.crosstab(
        df_comparison['rule_based'], 
        df_comparison['hmm_based'],
        margins=True
    )
    print(confusion)
    
    # Visualize comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Rule-based distribution
    rule_counts = df_comparison['rule_based'].value_counts()
    axes[0].bar(rule_counts.index, rule_counts.values, color='steelblue', alpha=0.7)
    axes[0].set_title('Rule-based Decisions', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Decision Type')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)
    
    # HMM-based distribution
    hmm_counts = df_comparison['hmm_based'].value_counts()
    axes[1].bar(hmm_counts.index, hmm_counts.values, color='coral', alpha=0.7)
    axes[1].set_title('HMM-based Decisions', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Decision Type')
    axes[1].set_ylabel('Count')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return df_comparison
