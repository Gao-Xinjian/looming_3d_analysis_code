"""
分层行为分析实现
Hierarchical Behavioral Analysis Implementation

三层架构:
1. Frame-level: 原始运动特征
2. Bout-level: 行为状态片段
3. Trial-level: 整体决策模式
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from hmmlearn import hmm
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Layer 1: Frame-level特征提取 (已有，复用现有代码)
# ============================================================================


# ============================================================================
# Layer 2: Bout-level分析 - 三种方法
# ============================================================================

class BoutDetector:
    """行为Bout检测器基类"""
    
    def __init__(self, method='hmm', n_states=3, min_bout_length=3):
        """
        Args:
            method: 'hmm', 'changepoint', 'clustering'
            n_states: 隐藏状态数量
            min_bout_length: 最小bout长度（帧数）
        """
        self.method = method
        self.n_states = n_states
        self.min_bout_length = min_bout_length
        self.model = None
        self.scaler = StandardScaler()
        
    def fit(self, all_observations):
        """训练模型"""
        if self.method == 'hmm':
            return self._fit_hmm(all_observations)
        elif self.method == 'clustering':
            return self._fit_clustering(all_observations)
        else:
            raise ValueError(f"Method {self.method} not implemented")
    
    def _fit_hmm(self, all_observations):
        """使用改进的HMM"""
        print(f"\n{'='*60}")
        print(f"Training HMM with {self.n_states} states")
        print(f"{'='*60}")
        
        # 1. 标准化所有特征
        X_all = np.vstack(all_observations)
        X_scaled = self.scaler.fit_transform(X_all)
        
        # 2. 重建序列长度
        lengths = [obs.shape[0] for obs in all_observations]
        
        # 3. 分段标准化后的数据
        scaled_observations = []
        start_idx = 0
        for length in lengths:
            scaled_observations.append(X_scaled[start_idx:start_idx+length])
            start_idx += length
        
        print(f"Training data:")
        print(f"  Total frames: {X_scaled.shape[0]}")
        print(f"  Sequences: {len(lengths)}")
        print(f"  Avg length: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
        print(f"  Feature dims: {X_scaled.shape[1]}")
        
        # 4. 训练HMM（使用对角协方差）
        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type='diag',  # 对角协方差，更稳定
            n_iter=200,
            tol=1e-4,
            init_params='stmc',
            random_state=42,
            verbose=False
        )
        
        self.model.fit(X_scaled, lengths)
        
        print(f"\nTraining completed:")
        print(f"  Converged: {self.model.monitor_.converged}")
        print(f"  Iterations: {self.model.monitor_.iter}")
        print(f"  Log-likelihood: {self.model.score(X_scaled, lengths):.2f}")
        
        # 5. 分析各状态特征（反标准化）
        self._interpret_states()
        
        return self
    
    def _fit_clustering(self, all_observations):
        """使用GMM聚类"""
        print(f"\n{'='*60}")
        print(f"Training GMM with {self.n_states} clusters")
        print(f"{'='*60}")
        
        # 标准化
        X_all = np.vstack(all_observations)
        X_scaled = self.scaler.fit_transform(X_all)
        
        # GMM聚类
        self.model = GaussianMixture(
            n_components=self.n_states,
            covariance_type='full',
            max_iter=200,
            random_state=42
        )
        self.model.fit(X_scaled)
        
        print(f"Training completed:")
        print(f"  Converged: {self.model.converged_}")
        print(f"  BIC: {self.model.bic(X_scaled):.2f}")
        
        return self
    
    def predict(self, observation):
        """预测状态序列"""
        # 标准化
        X_scaled = self.scaler.transform(observation)
        
        # 预测
        if self.method == 'hmm':
            states = self.model.predict(X_scaled)
        elif self.method == 'clustering':
            states = self.model.predict(X_scaled)
        else:
            raise ValueError(f"Method {self.method} not implemented")
        
        return states
    
    def extract_bouts(self, states):
        """从状态序列提取bout"""
        bouts = []
        
        if len(states) == 0:
            return bouts
        
        current_state = states[0]
        start_frame = 0
        
        for i, state in enumerate(states[1:], 1):
            if state != current_state:
                # 状态改变，记录上一个bout
                duration = i - start_frame
                if duration >= self.min_bout_length:
                    bouts.append({
                        'state': int(current_state),
                        'start': start_frame,
                        'end': i,
                        'duration': duration
                    })
                
                current_state = state
                start_frame = i
        
        # 最后一个bout
        duration = len(states) - start_frame
        if duration >= self.min_bout_length:
            bouts.append({
                'state': int(current_state),
                'start': start_frame,
                'end': len(states),
                'duration': duration
            })
        
        return bouts
    
    def _interpret_states(self):
        """解释各状态的特征（仅HMM）"""
        if self.method != 'hmm' or self.model is None:
            return
        
        print(f"\nState interpretations (standardized means):")
        feature_names = ['Speed', 'Dist2Nest', 'Dist2Looming', 'Angle2Looming']
        
        for state in range(self.n_states):
            mean = self.model.means_[state]
            print(f"  State {state}:")
            for fname, val in zip(feature_names, mean):
                print(f"    {fname}: {val:+.2f}")


def prepare_continuous_features(all_features):
    """
    准备纯连续特征（不包含behavior code）
    
    Returns:
        observations: list of (n_frames, 4) arrays
        metadata: 对应的session/trial/looming信息
    """
    observations = []
    metadata = []
    
    for sess_id, trials in all_features.items():
        for trial_id, loomings in trials.items():
            for looming_id, features in loomings.items():
                # 只取4个连续特征
                obs = np.column_stack([
                    features['speed'],
                    features['dist2nest'],
                    features['dist2looming'],
                    features['angle2looming'],
                ])
                
                observations.append(obs)
                metadata.append({
                    'sess_id': sess_id,
                    'trial_id': trial_id,
                    'looming_id': looming_id,
                    'n_frames': obs.shape[0]
                })
    
    return observations, metadata


# ============================================================================
# Layer 3: Trial-level特征提取
# ============================================================================

def extract_trial_features(observation, states, bouts, nest_series, feature_dict, fps=30):
    """
    从bout序列和原始特征提取trial-level特征
    
    Args:
        observation: (n_frames, 4) 标准化前的特征
        states: (n_frames,) 预测的状态序列
        bouts: list of bout dicts
        nest_series: (n_frames,) nest zone标记
        feature_dict: 原始特征字典（包含所有特征）
        fps: 帧率
        
    Returns:
        trial_features: dict of trial-level特征
    """
    n_frames = observation.shape[0]
    duration_sec = n_frames / fps
    
    # 基础统计
    speeds = observation[:, 0]
    dist2nest = observation[:, 1]
    dist2looming = observation[:, 2]
    
    # === Pathway特征（中间过程） ===
    
    # 初始反应
    initial_state = states[0] if len(states) > 0 else -1
    
    # 状态切换统计
    n_state_switches = np.sum(np.diff(states) != 0)
    
    # 各状态时间占比
    state_counts = Counter(states)
    state_proportions = {f'state_{s}_prop': count/n_frames 
                        for s, count in state_counts.items()}
    
    # Bout统计
    n_bouts = len(bouts)
    bout_durations = [b['duration'] for b in bouts]
    mean_bout_duration = np.mean(bout_durations) / fps if bout_durations else 0
    
    # 各状态的bout数量
    bout_state_counts = Counter([b['state'] for b in bouts])
    
    # 运动特征
    mean_speed = np.mean(speeds)
    max_speed = np.max(speeds)
    speed_var = np.var(speeds)
    
    # 高速运动时间比例（定义为 > 10 cm/s）
    high_speed_ratio = np.sum(speeds > 10) / n_frames
    
    # 到nest的距离变化
    initial_dist2nest = dist2nest[0]
    final_dist2nest = dist2nest[-1]
    min_dist2nest = np.min(dist2nest)
    dist2nest_change = initial_dist2nest - final_dist2nest  # 正值表示靠近
    
    # 路径效率（理想距离 vs 实际移动距离）
    displacement = np.abs(initial_dist2nest - final_dist2nest)
    # 累计移动距离（近似）
    frame_displacements = np.abs(np.diff(dist2nest))
    total_distance = np.sum(frame_displacements)
    path_efficiency = displacement / (total_distance + 1e-6)
    
    # 到达nest的延迟
    # 定义"到达nest"为距离 < 5cm
    nest_threshold = 5.0
    reached_nest_frames = np.where(dist2nest < nest_threshold)[0]
    if len(reached_nest_frames) > 0:
        latency_to_nest = reached_nest_frames[0] / fps
    else:
        latency_to_nest = duration_sec  # 未到达
    
    # === Outcome特征（最终结果） ===
    
    # 最终状态
    final_state = states[-1] if len(states) > 0 else -1
    
    # 是否到达nest
    reached_nest = final_dist2nest < nest_threshold
    
    # Nest停留时间
    if hasattr(nest_series, 'values'):
        nest_array = nest_series.values
    else:
        nest_array = nest_series
    time_in_nest = np.sum(nest_array) / fps
    
    # 最终是否在nest中
    final_in_nest = nest_array[-1] if len(nest_array) > 0 else False
    
    # 成功逃脱定义：到达nest且停留 > 0.5秒
    success_escape = reached_nest and (time_in_nest > 0.5)
    
    # === 汇总所有特征 ===
    trial_features = {
        # Metadata
        'duration_sec': duration_sec,
        'n_frames': n_frames,
        
        # Pathway - 初始反应
        'initial_state': initial_state,
        
        # Pathway - 状态转换
        'n_state_switches': n_state_switches,
        'n_bouts': n_bouts,
        'mean_bout_duration': mean_bout_duration,
        
        # Pathway - 状态分布
        **state_proportions,
        
        # Pathway - 各状态bout数
        **{f'n_bouts_state_{s}': count for s, count in bout_state_counts.items()},
        
        # Pathway - 运动特征
        'mean_speed': mean_speed,
        'max_speed': max_speed,
        'speed_variance': speed_var,
        'high_speed_ratio': high_speed_ratio,
        
        # Pathway - 空间特征
        'initial_dist2nest': initial_dist2nest,
        'dist2nest_change': dist2nest_change,
        'min_dist2nest': min_dist2nest,
        'path_efficiency': path_efficiency,
        
        # Pathway - 时间特征
        'latency_to_nest': latency_to_nest,
        'time_in_nest': time_in_nest,
        
        # Outcome
        'final_state': final_state,
        'final_dist2nest': final_dist2nest,
        'final_in_nest': bool(final_in_nest),
        'reached_nest': reached_nest,
        'success_escape': success_escape,
    }
    
    return trial_features


def batch_extract_trial_features(all_features, detector, nest_dict, fps=30):
    """
    批量提取所有trial的特征
    
    Args:
        all_features: 原始特征字典
        detector: 训练好的BoutDetector
        nest_dict: nest zone字典
        fps: 帧率
        
    Returns:
        trial_features_df: DataFrame包含所有trial的特征
        all_bouts_dict: 每个trial的bout序列
    """
    # 准备观测数据
    observations, metadata = prepare_continuous_features(all_features)
    
    # 存储结果
    all_trial_features = []
    all_bouts_dict = {}
    
    print(f"\n{'='*60}")
    print(f"Extracting trial-level features")
    print(f"{'='*60}")
    
    for obs, meta in zip(observations, metadata):
        sess_id = meta['sess_id']
        trial_id = meta['trial_id']
        looming_id = meta['looming_id']
        
        # 预测状态
        states = detector.predict(obs)
        
        # 提取bouts
        bouts = detector.extract_bouts(states)
        
        # 获取对应的nest series
        # 需要根据looming_id切分nest_series
        # 这里假设已经在all_features中有对应的nest数据
        feature_dict = all_features[sess_id][trial_id][looming_id]
        nest_series = feature_dict['in_nest']
        
        # 提取trial特征
        trial_feats = extract_trial_features(
            obs, states, bouts, nest_series, feature_dict, fps
        )
        
        # 添加metadata
        trial_feats.update({
            'sess_id': sess_id,
            'trial_id': trial_id,
            'looming_id': looming_id,
        })
        
        all_trial_features.append(trial_feats)
        
        # 保存bouts
        key = f"{sess_id}_{trial_id}_L{looming_id}"
        all_bouts_dict[key] = {
            'states': states,
            'bouts': bouts,
            'observation': obs
        }
        
        if len(all_trial_features) % 50 == 0:
            print(f"  Processed {len(all_trial_features)} trials...")
    
    print(f"Completed! Total trials: {len(all_trial_features)}")
    
    # 转换为DataFrame
    df = pd.DataFrame(all_trial_features)
    
    return df, all_bouts_dict


# ============================================================================
# 完整Pipeline
# ============================================================================

def run_hierarchical_analysis(all_features, nest_dict, method='hmm', n_states=3, fps=30):
    """
    运行完整的分层分析
    
    Args:
        all_features: 从prepare_observation_sequences提取的特征
        nest_dict: nest zone字典
        method: 'hmm' or 'clustering'
        n_states: 隐藏状态数量
        fps: 帧率
        
    Returns:
        detector: 训练好的BoutDetector
        trial_features_df: Trial-level特征DataFrame
        bouts_dict: 所有trial的bout信息
    """
    
    print(f"\n{'#'*60}")
    print(f"# Hierarchical Behavioral Analysis Pipeline")
    print(f"#   Method: {method.upper()}")
    print(f"#   States: {n_states}")
    print(f"{'#'*60}")
    
    # Step 1: 准备连续特征
    print(f"\nStep 1: Preparing continuous features...")
    observations, metadata = prepare_continuous_features(all_features)
    print(f"  {len(observations)} observation sequences prepared")
    
    # Step 2: 训练Bout检测器
    print(f"\nStep 2: Training bout detector...")
    detector = BoutDetector(method=method, n_states=n_states, min_bout_length=3)
    detector.fit(observations)
    
    # Step 3: 提取Trial特征
    print(f"\nStep 3: Extracting trial-level features...")
    trial_features_df, bouts_dict = batch_extract_trial_features(
        all_features, detector, nest_dict, fps
    )
    
    # Step 4: 总结
    print(f"\n{'='*60}")
    print(f"Analysis completed!")
    print(f"{'='*60}")
    print(f"Trial features extracted:")
    print(f"  Total trials: {len(trial_features_df)}")
    print(f"  Feature columns: {len(trial_features_df.columns)}")
    print(f"\nKey outcome statistics:")
    print(f"  Success escape rate: {trial_features_df['success_escape'].mean():.1%}")
    print(f"  Reached nest rate: {trial_features_df['reached_nest'].mean():.1%}")
    print(f"  Mean latency to nest: {trial_features_df['latency_to_nest'].mean():.2f}s")
    
    return detector, trial_features_df, bouts_dict


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == '__main__':
    """
    在您的notebook中使用：
    
    # 假设已经运行了之前的特征提取代码
    # all_features, behavior_encoder, feature_stats = prepare_observation_sequences(...)
    
    # 运行分层分析
    detector, trial_df, bouts_dict = run_hierarchical_analysis(
        all_features=all_features,
        nest_dict=nest_series_dict,
        method='hmm',      # 或 'clustering'
        n_states=3,        # 建议从3开始
        fps=fps
    )
    
    # 保存结果
    trial_df.to_csv('trial_features.csv', index=False)
    
    with open('bout_detector.pkl', 'wb') as f:
        pickle.dump(detector, f)
    
    with open('bouts_dict.pkl', 'wb') as f:
        pickle.dump(bouts_dict, f)
    
    # 查看trial特征
    print(trial_df.head())
    
    # 按组比较
    trial_df['group'] = trial_df['sess_id'].str[:2]  # G1, G2, ...
    trial_df.groupby('group')['success_escape'].mean()
    """
    pass
