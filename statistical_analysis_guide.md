# 后续统计分析指南
# Statistical Analysis Guide for Hierarchical Approach

## 总览
使用分层分析框架后，您可以在不同层级进行统计分析，从而全面描述小鼠的decision-making process。

---

## 1. Frame-level分析

### 1.1 轨迹分析
```python
# 绘制运动轨迹，按状态着色
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def plot_trajectory_with_states(coords, states, sess_id, trial_id):
    """绘制带状态标记的运动轨迹"""
    # 提取XY坐标（SpineM）
    spine_idx = joint_names.index('SpineM')
    xy = coords[:, spine_idx, :2]
    
    # 创建线段
    points = xy.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # 根据状态着色
    colors = ['blue', 'orange', 'green', 'red']
    state_colors = [colors[s] for s in states[:-1]]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    lc = LineCollection(segments, colors=state_colors, linewidths=2)
    ax.add_collection(lc)
    
    # 添加nest和looming位置
    nest_rect = plt.Rectangle((0, 0), 16.23, 22.65, 
                              fill=False, edgecolor='green', linewidth=2)
    ax.add_patch(nest_rect)
    ax.plot(25.0, 35.75, 'r*', markersize=20, label='Looming')
    
    ax.set_xlim(-5, 55)
    ax.set_ylim(-5, 55)
    ax.set_aspect('equal')
    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Y (cm)')
    ax.set_title(f'Trajectory: {sess_id} {trial_id}')
    ax.legend()
    
    return fig

# 使用示例
# fig = plot_trajectory_with_states(
#     coords=coord_dict[sess_id][trial_frames],
#     states=bouts_dict_hmm[trial_key]['states'],
#     sess_id='G1_M1',
#     trial_id='T1'
# )
```

### 1.2 瞬时速度分析
```python
# 按状态比较瞬时速度
import seaborn as sns

def analyze_speed_by_state(all_features, bouts_dict):
    """分析各状态的速度分布"""
    data_list = []
    
    for key, bout_info in bouts_dict.items():
        parts = key.split('_')
        sess_id = parts[0]
        trial_id = '_'.join(parts[1:-1])
        looming_id = int(parts[-1][1:])
        
        states = bout_info['states']
        features = all_features[sess_id][trial_id][looming_id]
        speeds = features['speed']
        
        for state in range(3):  # 假设3个状态
            state_speeds = speeds[states == state]
            for speed in state_speeds:
                data_list.append({
                    'state': state,
                    'speed': speed,
                    'sess_id': sess_id
                })
    
    df = pd.DataFrame(data_list)
    
    # 可视化
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(data=df, x='state', y='speed', ax=ax)
    ax.set_xlabel('State')
    ax.set_ylabel('Speed (cm/s)')
    ax.set_title('Speed Distribution by State')
    
    # 统计检验
    from scipy import stats
    states_list = [df[df['state']==s]['speed'].values for s in range(3)]
    f_stat, p_value = stats.f_oneway(*states_list)
    print(f"One-way ANOVA: F={f_stat:.2f}, p={p_value:.4f}")
    
    return df, fig

# 使用
# speed_df, fig = analyze_speed_by_state(all_features, bouts_dict_hmm)
```

---

## 2. Bout-level分析

### 2.1 状态转移分析
```python
def analyze_state_transitions(bouts_dict, n_states=3):
    """分析状态转移矩阵"""
    # 初始化转移计数矩阵
    trans_count = np.zeros((n_states, n_states))
    
    for key, bout_info in bouts_dict.items():
        bouts = bout_info['bouts']
        
        for i in range(len(bouts) - 1):
            from_state = bouts[i]['state']
            to_state = bouts[i+1]['state']
            trans_count[from_state, to_state] += 1
    
    # 转换为概率
    trans_prob = trans_count / trans_count.sum(axis=1, keepdims=True)
    
    # 可视化
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(trans_prob, cmap='Blues', vmin=0, vmax=1)
    
    # 添加数值标签
    for i in range(n_states):
        for j in range(n_states):
            text = ax.text(j, i, f'{trans_prob[i, j]:.2f}',
                          ha="center", va="center", color="black")
    
    ax.set_xticks(range(n_states))
    ax.set_yticks(range(n_states))
    ax.set_xlabel('To State')
    ax.set_ylabel('From State')
    ax.set_title('State Transition Probability Matrix')
    plt.colorbar(im, ax=ax)
    
    return trans_prob, fig

# 使用
# trans_prob, fig = analyze_state_transitions(bouts_dict_hmm, n_states=3)
```

### 2.2 Bout持续时间分析
```python
def analyze_bout_durations(bouts_dict, fps=30):
    """分析bout持续时间"""
    data_list = []
    
    for key, bout_info in bouts_dict.items():
        parts = key.split('_')
        sess_id = parts[0]
        group = sess_id[:2]
        
        for bout in bout_info['bouts']:
            data_list.append({
                'state': bout['state'],
                'duration_sec': bout['duration'] / fps,
                'group': group,
                'sess_id': sess_id
            })
    
    df = pd.DataFrame(data_list)
    
    # 按状态和组可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：按状态
    sns.boxplot(data=df, x='state', y='duration_sec', ax=axes[0])
    axes[0].set_xlabel('State')
    axes[0].set_ylabel('Bout Duration (s)')
    axes[0].set_title('Bout Duration by State')
    
    # 右图：按组
    sns.boxplot(data=df, x='group', y='duration_sec', hue='state', ax=axes[1])
    axes[1].set_xlabel('Group')
    axes[1].set_ylabel('Bout Duration (s)')
    axes[1].set_title('Bout Duration by Group and State')
    
    plt.tight_layout()
    
    return df, fig

# 使用
# bout_dur_df, fig = analyze_bout_durations(bouts_dict_hmm, fps=fps)
```

---

## 3. Trial-level分析（重点）

### 3.1 组间比较（主要分析）
```python
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def compare_groups(trial_df):
    """比较不同looming间隔组的表现"""
    
    # 添加group列
    trial_df['group'] = trial_df['sess_id'].str[:2]
    
    # 定义关键指标
    key_metrics = [
        'success_escape',      # 成功逃脱率
        'latency_to_nest',     # 到达nest延迟
        'mean_speed',          # 平均速度
        'n_state_switches',    # 状态切换次数
        'path_efficiency',     # 路径效率
        'time_in_nest',        # nest停留时间
    ]
    
    results = {}
    
    for metric in key_metrics:
        print(f"\n{'='*60}")
        print(f"Metric: {metric}")
        print(f"{'='*60}")
        
        # 描述性统计
        group_stats = trial_df.groupby('group')[metric].agg(['mean', 'std', 'count'])
        print("\nDescriptive statistics:")
        print(group_stats)
        
        # One-way ANOVA
        groups = [trial_df[trial_df['group']==g][metric].values 
                 for g in ['G1', 'G2', 'G3', 'G4', 'G5']]
        f_stat, p_value = stats.f_oneway(*groups)
        print(f"\nOne-way ANOVA: F={f_stat:.3f}, p={p_value:.4f}")
        
        # Post-hoc Tukey HSD (如果显著)
        if p_value < 0.05:
            print("\nPost-hoc Tukey HSD:")
            tukey = pairwise_tukeyhsd(trial_df[metric], trial_df['group'])
            print(tukey)
        
        results[metric] = {
            'group_stats': group_stats,
            'anova_f': f_stat,
            'anova_p': p_value
        }
    
    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(key_metrics):
        sns.boxplot(data=trial_df, x='group', y=metric, ax=axes[i])
        axes[i].set_title(f'{metric}\n(F={results[metric]["anova_f"]:.2f}, p={results[metric]["anova_p"]:.4f})')
        axes[i].set_xlabel('Group')
        axes[i].set_ylabel(metric)
    
    plt.tight_layout()
    
    return results, fig

# 使用
# results, fig = compare_groups(trial_df_hmm)
```

### 3.2 Pathway vs Outcome分析
```python
def analyze_pathway_outcome_relationship(trial_df):
    """分析中间过程（pathway）与最终结果（outcome）的关系"""
    
    # Pathway特征
    pathway_features = [
        'n_state_switches',
        'mean_speed',
        'path_efficiency',
        'latency_to_nest'
    ]
    
    # Outcome
    outcome = 'success_escape'
    
    # 按outcome分组比较pathway
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, feature in enumerate(pathway_features):
        sns.violinplot(data=trial_df, x=outcome, y=feature, ax=axes[i])
        axes[i].set_xlabel('Success Escape')
        axes[i].set_ylabel(feature)
        axes[i].set_title(f'{feature} by Outcome')
        
        # t检验
        success = trial_df[trial_df[outcome]==1][feature].values
        fail = trial_df[trial_df[outcome]==0][feature].values
        t_stat, p_value = stats.ttest_ind(success, fail)
        axes[i].text(0.5, 0.95, f't={t_stat:.2f}, p={p_value:.4f}',
                    transform=axes[i].transAxes, ha='center', va='top')
    
    plt.tight_layout()
    
    return fig

# 使用
# fig = analyze_pathway_outcome_relationship(trial_df_hmm)
```

### 3.3 Logistic回归：预测成功逃脱
```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

def predict_success_escape(trial_df):
    """用pathway特征预测success escape"""
    
    # 选择特征
    features = [
        'initial_state',
        'n_state_switches',
        'mean_speed',
        'latency_to_nest',
        'path_efficiency',
    ]
    
    # 准备数据
    X = trial_df[features].values
    y = trial_df['success_escape'].values.astype(int)
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 训练模型
    model = LogisticRegression(random_state=42)
    
    # 交叉验证
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='roc_auc')
    print(f"Cross-validation AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    # 拟合完整模型
    model.fit(X_scaled, y)
    
    # 特征重要性
    importance_df = pd.DataFrame({
        'feature': features,
        'coefficient': model.coef_[0]
    }).sort_values('coefficient', key=abs, ascending=False)
    
    print("\nFeature importance:")
    print(importance_df)
    
    # 可视化
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(importance_df['feature'], importance_df['coefficient'])
    ax.set_xlabel('Coefficient')
    ax.set_title('Logistic Regression: Predicting Success Escape')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    
    return model, importance_df, fig

# 使用
# model, importance, fig = predict_success_escape(trial_df_hmm)
```

---

## 4. Looming-level分析（如有5个looming）

### 4.1 习惯化分析
```python
def analyze_habituation(trial_df):
    """分析looming间的习惯化效应（仅G1-G4有多个looming）"""
    
    # 过滤出有5个looming的组
    multi_looming_groups = ['G1', 'G2', 'G3', 'G4']
    df_multi = trial_df[trial_df['group'].isin(multi_looming_groups)].copy()
    
    # 按looming编号分组
    metrics = ['success_escape', 'latency_to_nest', 'mean_speed']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for i, metric in enumerate(metrics):
        # 计算每个looming的平均值
        looming_avg = df_multi.groupby(['looming_id', 'group'])[metric].mean().reset_index()
        
        # 按组绘制
        for group in multi_looming_groups:
            data = looming_avg[looming_avg['group']==group]
            axes[i].plot(data['looming_id'], data[metric], 
                        marker='o', label=group)
        
        axes[i].set_xlabel('Looming Number')
        axes[i].set_ylabel(metric)
        axes[i].set_title(f'{metric} across Loomings')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 统计检验：是否有显著的looming效应
    print("Repeated measures ANOVA (within each group):")
    for group in multi_looming_groups:
        df_group = df_multi[df_multi['group']==group]
        
        for metric in metrics:
            # 简化：用Kruskal-Wallis检验
            looming_groups = [df_group[df_group['looming_id']==i][metric].values 
                            for i in range(5)]
            h_stat, p_value = stats.kruskal(*looming_groups)
            print(f"  {group} - {metric}: H={h_stat:.2f}, p={p_value:.4f}")
    
    return fig

# 使用
# fig = analyze_habituation(trial_df_hmm)
```

---

## 5. 混合效应模型（高级）

### 5.1 层级模型
```python
import statsmodels.formula.api as smf

def fit_mixed_model(trial_df):
    """拟合混合效应模型，考虑session和trial的层级结构"""
    
    # 准备数据
    df = trial_df.copy()
    df['group'] = df['sess_id'].str[:2]
    
    # 模型：success_escape ~ group + (1|sess_id) + (1|trial_id)
    # 这里用更简单的版本
    
    model = smf.mixedlm(
        "success_escape ~ C(group) + mean_speed + path_efficiency",
        data=df,
        groups=df["sess_id"]
    )
    
    result = model.fit()
    print(result.summary())
    
    return result

# 使用
# result = fit_mixed_model(trial_df_hmm)
```

---

## 6. 综合报告生成

### 6.1 自动生成分析报告
```python
def generate_analysis_report(trial_df, bouts_dict, output_dir='analysis_results'):
    """生成完整的分析报告"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 描述性统计
    trial_df.to_csv(f'{output_dir}/trial_features.csv', index=False)
    
    # 2. 组间比较
    results, fig = compare_groups(trial_df)
    fig.savefig(f'{output_dir}/group_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # 3. Pathway-Outcome关系
    fig = analyze_pathway_outcome_relationship(trial_df)
    fig.savefig(f'{output_dir}/pathway_outcome.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # 4. 预测模型
    model, importance, fig = predict_success_escape(trial_df)
    fig.savefig(f'{output_dir}/prediction_model.png', dpi=300, bbox_inches='tight')
    importance.to_csv(f'{output_dir}/feature_importance.csv', index=False)
    plt.close(fig)
    
    # 5. 习惯化分析
    fig = analyze_habituation(trial_df)
    fig.savefig(f'{output_dir}/habituation.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # 6. 状态转移
    trans_prob, fig = analyze_state_transitions(bouts_dict)
    fig.savefig(f'{output_dir}/state_transitions.png', dpi=300, bbox_inches='tight')
    np.save(f'{output_dir}/transition_matrix.npy', trans_prob)
    plt.close(fig)
    
    print(f"\n✓ Analysis report generated in: {output_dir}/")

# 使用
# generate_analysis_report(trial_df_hmm, bouts_dict_hmm, output_dir='hmm_analysis_results')
```

---

## 总结

### 分析流程
1. **Frame-level**: 可视化轨迹、分析瞬时运动特征
2. **Bout-level**: 状态转移矩阵、bout持续时间
3. **Trial-level**: 
   - 组间比较（One-way ANOVA + post-hoc）
   - Pathway vs Outcome
   - 预测模型（Logistic回归）
4. **Looming-level**: 习惯化效应

### 核心优势
- ✅ 多层级分析，全面描述decision-making
- ✅ Pathway和Outcome分离，便于因果推断
- ✅ 适用于传统统计方法（ANOVA, regression）
- ✅ 结果易于解释和发表

### 推荐的分析顺序
1. 先看Trial-level的组间比较 → 确定哪些指标显著
2. 分析Pathway-Outcome关系 → 理解决策过程
3. 查看Bout-level的状态转移 → 揭示行为动力学
4. 可视化Frame-level轨迹 → 提供具体例子

---

## 下一步
如需帮助实现任何特定分析，请告诉我！
