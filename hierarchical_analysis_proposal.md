# 分层行为分析框架 (Hierarchical Behavioral Analysis)

## 总体思路
将复杂的decision-making过程分解为多个层级，每个层级关注不同的时间尺度和抽象程度。

## 三层分析架构

### Layer 1: Frame-level (底层特征)
**目标**: 捕捉瞬时状态和运动特征  
**时间尺度**: 单帧 (~33ms @ 30fps)  
**方法**: 
- 直接使用提取的连续特征（速度、距离、角度）
- 不需要HMM，这是raw observations

**输出**:
- 运动轨迹
- 空间位置序列
- 瞬时速度/加速度

---

### Layer 2: Bout-level (中层模式)
**目标**: 识别行为片段(bouts)和状态转换  
**时间尺度**: 几百毫秒到几秒  
**方法选项**:

#### 选项A: 改进的HMM（推荐）
```python
# 只使用连续特征，不包含behavior code
features = [
    'speed',              # 速度
    'dist2nest',          # 到nest距离
    'dist2looming',       # 到looming距离  
    'angle2looming',      # 朝向looming的角度
]

# 简化隐藏状态定义（3-4个状态即可）
states = [
    'Exploration',  # 探索：低速，远离nest和looming
    'Escape',       # 逃跑：高速，朝向nest
    'Freezing',     # 冻结：极低速，位置固定
    # 'Approach'    # 可选：如果有接近行为
]
```

#### 选项B: Change-Point Detection
```python
# 使用ruptures库检测行为切换点
import ruptures as rpt

# 基于多变量特征检测change points
algo = rpt.Pelt(model="rbf").fit(features)
change_points = algo.predict(pen=10)

# 每个segment就是一个bout
```

#### 选项C: Behavioral Clustering
```python
# 用HDBSCAN或GMM对frame-level特征聚类
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=4, covariance_type='full')
frame_states = gmm.fit_predict(features)

# 相邻相同状态的连续帧组成bout
```

**输出**:
- Bout序列：[(state, start_frame, end_frame, duration), ...]
- 状态转移矩阵
- Bout持续时间分布

---

### Layer 3: Trial-level (高层决策)
**目标**: 描述整体decision pattern和outcome  
**时间尺度**: 整个trial (几秒到几十秒)  
**方法**:

#### Step 1: 定义Trial-level特征
基于Layer 2的bout序列，提取：

```python
trial_features = {
    # Pathway特征（中间过程）
    'initial_response': first_bout_state,           # 初始反应
    'n_state_switches': count_state_transitions(),   # 状态切换次数
    'escape_bout_count': count_escape_bouts(),       # 逃跑bout数量
    'freeze_bout_count': count_freeze_bouts(),       # 冻结bout数量
    'total_escape_time': sum_escape_duration(),      # 总逃跑时间
    'mean_escape_speed': mean_speed_in_escape(),     # 逃跑平均速度
    'latency_to_nest': time_to_reach_nest(),        # 到达nest延迟
    'path_efficiency': direct_dist / actual_dist,    # 路径效率
    
    # Outcome特征（最终结果）
    'final_state': last_bout_state,                  # 最终状态
    'reached_nest': bool(in_nest_at_end),           # 是否到达nest
    'time_in_nest': total_time_in_nest_zone,        # nest停留时间
    'success_escape': reached_nest and stayed,       # 成功逃脱
}
```

#### Step 2: Trial分类
```python
# 基于trial特征分类decision pattern
from sklearn.cluster import KMeans

trial_patterns = KMeans(n_clusters=3).fit_predict(trial_feature_matrix)

# 或使用规则定义
def classify_trial(features):
    if features['reached_nest'] and features['latency_to_nest'] < 1.0:
        return 'Fast_Escape'
    elif features['freeze_bout_count'] > 2:
        return 'Freeze_dominant'
    elif features['n_state_switches'] > 5:
        return 'Hesitant'
    else:
        return 'Other'
```

---

## 实现建议

### 步骤1: 修复当前HMM
如果继续使用HMM，做以下改动：

```python
def prepare_hmm_observations_v2(all_features):
    """只使用连续特征，标准化处理"""
    observations = []
    
    for sess_id, trials in all_features.items():
        for trial_id, loomings in trials.items():
            for looming_id, features in loomings.items():
                # 只取连续特征
                obs = np.column_stack([
                    features['speed'],
                    features['dist2nest'],
                    features['dist2looming'],
                    features['angle2looming'],
                ])
                
                # 标准化（z-score）
                obs = (obs - obs.mean(axis=0)) / (obs.std(axis=0) + 1e-8)
                
                observations.append(obs)
    
    return observations

# 使用更少的状态
model = hmm.GaussianHMM(
    n_components=3,  # 减少到3-4个状态
    covariance_type='diag',  # 简化协方差结构
    n_iter=200,      # 增加迭代次数
    tol=1e-4,        # 调整收敛阈值
    init_params='stmc',  # 初始化参数
    random_state=42
)
```

### 步骤2: 从Bout序列提取Trial特征
```python
def extract_trial_features(bout_sequence, coord_sequence, nest_sequence):
    """从bout序列提取trial-level特征"""
    # 这是连接Layer 2和Layer 3的关键
    # 实现见下方完整代码
    pass
```

### 步骤3: 多层级统计分析
```python
# Frame-level: 时序分析
plot_trajectory(coords, bouts)

# Bout-level: 状态转移
transition_matrix = calculate_transition_prob(bout_sequences)

# Trial-level: 组间比较
compare_groups(trial_features, groups=['G1', 'G2', 'G3', 'G4', 'G5'])
```

---

## 为什么这个框架更好？

1. **解耦观测和状态**: 不用behavior code作为HMM输入，避免循环依赖
2. **层级清晰**: 每层关注不同抽象级别，便于解释
3. **灵活性**: 可以在每层选择最合适的方法（HMM, change-point, clustering）
4. **可解释性**: Pathway和Outcome分离，便于分析decision-making过程
5. **统计友好**: Trial-level特征可以用传统统计方法（ANOVA, regression）

---

## Looming-level分析
如果有5个looming，可以增加一层：

```python
# 对比同一trial内不同looming的反应
looming_adaptation = {
    'habituation': response[4] < response[0],  # 是否习惯化
    'sensitization': response[4] > response[0],  # 是否敏感化
    'response_sequence': [r for r in responses],  # 反应序列
}
```

---

## 下一步
我可以帮您实现以下任一部分：
1. 修复当前HMM代码（标准化、减少状态数）
2. 实现change-point detection方法
3. 编写bout提取和trial特征计算函数
4. 建立完整的分层分析pipeline

您想从哪个开始？
