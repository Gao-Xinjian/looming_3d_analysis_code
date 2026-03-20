#!/usr/bin/env python3
"""
Script to add HMM decision analysis cells to analyze_data_3d.ipynb
"""

import json

# Read the notebook
notebook_path = '/home/gxj/Desktop/gxj/code/lst_3d_code/analyze_data_3d.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Find the cell with "#### decision-making analysis: HMM"
hmm_cell_index = None
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'markdown':
        if '#### decision-making analysis: HMM' in ''.join(cell['source']):
            hmm_cell_index = i
            break

if hmm_cell_index is None:
    print("Could not find HMM section!")
    exit(1)

print(f"Found HMM section at cell index {hmm_cell_index}")

# Replace the empty code cell after HMM title with new cells
insert_position = hmm_cell_index + 2  # After the HMM markdown cell and empty code cell

# New cells to insert
new_cells = [
    # Cell 1: Explanation
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "**HMM决策分析说明**\n",
            "\n",
            "Hidden Markov Model (HMM) 用于从行为观测序列推断小鼠的决策状态。\n",
            "\n",
            "**核心概念：**\n",
            "- **观测 (Observations)**：直接观察到的数据\n",
            "  - 原始行为标签 (walking, grooming, rearing, etc.)\n",
            "  - 运动学特征 (速度, 加速度, 与nest距离等)\n",
            "  \n",
            "- **隐藏状态 (Hidden States)**：想要推断的决策状态\n",
            "  - **E (Escape)**: 逃跑决策\n",
            "  - **F (Freeze)**: 冻结决策  \n",
            "  - **R (Retreat)**: 撤退决策\n",
            "  - **I (Ignore)**: 忽略决策\n",
            "  \n",
            "**HMM vs 基于规则的方法：**\n",
            "- 规则方法：手动定义\"如果出现jumping则判定为E\"\n",
            "- HMM方法：自动学习\"出现jumping时，有X%概率是E状态，Y%概率是F状态\"\n",
            "- HMM考虑时间序列和状态转移，更灵活\n",
            "\n",
            "**分析层次：**\n",
            "1. **Looming decision**: 每个looming刺激的决策状态序列\n",
            "2. **Trial decision**: 整个trial的总体决策（基于looming decisions汇总）"
        ]
    },
    # Cell 2: Import module
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Import HMM analysis module\n",
            "import sys\n",
            "sys.path.append('/home/gxj/Desktop/gxj/code/lst_3d_code')\n",
            "from hmm_decision_analysis import *\n",
            "\n",
            "print(\"HMM analysis module imported successfully\")"
        ]
    },
    # Cell 3: Extract features
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Step 1: Extract features for HMM from all sessions\n",
            "print(\"Extracting features for HMM...\\n\")\n",
            "\n",
            "all_features = {}\n",
            "all_looming_sequences = {}\n",
            "\n",
            "for sess_id in lsf_dict.keys():\n",
            "    print(f\"Processing {sess_id}...\")\n",
            "    \n",
            "    # Extract features\n",
            "    features = extract_features_for_hmm(\n",
            "        sess_id,\n",
            "        bhvr_tuples_dict[sess_id],\n",
            "        coord_dict[sess_id],\n",
            "        nest_tuples_dict[sess_id],\n",
            "        joint_names,\n",
            "        fps=fps\n",
            "    )\n",
            "    all_features[sess_id] = features\n",
            "    \n",
            "    # Prepare observation sequences\n",
            "    looming_seqs, behavior_encoder = prepare_hmm_observation_sequences(\n",
            "        sess_id,\n",
            "        lsf_dict[sess_id],\n",
            "        features,\n",
            "        fps=fps\n",
            "    )\n",
            "    all_looming_sequences[sess_id] = looming_seqs\n",
            "\n",
            "print(f\"\\nFeature extraction completed for {len(all_features)} sessions\")"
        ]
    },
    # Cell 4: Prepare training data
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Step 2: Collect all observation sequences for training\n",
            "all_obs_for_training = []\n",
            "\n",
            "for sess_id, trials in all_looming_sequences.items():\n",
            "    for trial_id, loomings in trials.items():\n",
            "        for looming_id, data in loomings.items():\n",
            "            all_obs_for_training.append(data['observations'])\n",
            "\n",
            "print(f\"Collected {len(all_obs_for_training)} observation sequences for training\")\n",
            "print(f\"Feature dimensions: {all_obs_for_training[0].shape[1]}\")\n",
            "print(f\"  [0]: Behavior code\")\n",
            "print(f\"  [1]: Velocity (mm/s)\")\n",
            "print(f\"  [2]: Acceleration\")\n",
            "print(f\"  [3]: Distance to nest (mm)\")"
        ]
    },
    # Cell 5: Train HMM
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Step 3: Train HMM model with 4 hidden states (E, F, R, I)\n",
            "n_decision_states = 4\n",
            "hmm_model = train_hmm_model(all_obs_for_training, n_states=n_decision_states, n_iter=100)\n",
            "\n",
            "# Save the model\n",
            "with open('hmm_decision_model.pkl', 'wb') as f:\n",
            "    pickle.dump(hmm_model, f)\n",
            "\n",
            "print(\"\\nHMM model trained and saved as 'hmm_decision_model.pkl'\")"
        ]
    },
    # Cell 6: Analyze states
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Step 4: Analyze HMM states to understand what each represents\n",
            "state_features = analyze_hmm_states(hmm_model, all_looming_sequences, n_decision_states)"
        ]
    },
    # Cell 7: State mapping
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Step 5: Create state mapping based on analysis above\n",
            "# ⚠️ YOU NEED TO ADJUST THIS MAPPING based on the output from previous cell\n",
            "# Look at velocity, acceleration, and distance patterns to assign states\n",
            "\n",
            "state_to_decision = {\n",
            "    0: 'I',  # Example: State 0 → Ignore (adjust based on your data)\n",
            "    1: 'F',  # Example: State 1 → Freeze\n",
            "    2: 'R',  # Example: State 2 → Retreat\n",
            "    3: 'E'   # Example: State 3 → Escape\n",
            "}\n",
            "\n",
            "print(\"State to decision mapping:\")\n",
            "for state, decision in state_to_decision.items():\n",
            "    print(f\"  State {state} → {decision}\")\n",
            "\n",
            "print(\"\\n⚠️ Please verify this mapping based on the state characteristics above!\")"
        ]
    },
    # Cell 8: Predict looming decisions
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Step 6: Predict looming-level decisions using HMM\n",
            "print(\"Predicting looming-level decision states using HMM...\\n\")\n",
            "\n",
            "hmm_looming_decisions = {}\n",
            "hmm_looming_states = {}  # Store raw state sequences\n",
            "\n",
            "for sess_id, trials in all_looming_sequences.items():\n",
            "    hmm_looming_decisions[sess_id] = {}\n",
            "    hmm_looming_states[sess_id] = {}\n",
            "    \n",
            "    for trial_id, loomings in trials.items():\n",
            "        trial_decisions = []\n",
            "        trial_states = []\n",
            "        \n",
            "        for looming_id in range(len(loomings)):\n",
            "            if looming_id in loomings:\n",
            "                obs = loomings[looming_id]['observations']\n",
            "                \n",
            "                # Predict states\n",
            "                states = predict_decision_states(hmm_model, obs)\n",
            "                trial_states.append(states)\n",
            "                \n",
            "                # Interpret state with mapping\n",
            "                decision = interpret_hmm_states(states, state_to_decision)\n",
            "                trial_decisions.append(decision)\n",
            "        \n",
            "        hmm_looming_decisions[sess_id][trial_id] = trial_decisions\n",
            "        hmm_looming_states[sess_id][trial_id] = trial_states\n",
            "\n",
            "print(f\"HMM looming-level predictions completed\")\n",
            "\n",
            "# Show example predictions\n",
            "example_sess = list(hmm_looming_decisions.keys())[0]\n",
            "example_trial = list(hmm_looming_decisions[example_sess].keys())[0]\n",
            "print(f\"\\nExample - {example_trial}: {hmm_looming_decisions[example_sess][example_trial]}\")"
        ]
    },
    # Cell 9: Calculate trial decisions
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Step 7: Calculate trial-level decisions from looming decisions\n",
            "print(\"Calculating trial-level decisions...\\n\")\n",
            "\n",
            "hmm_trial_decisions = {}\n",
            "\n",
            "for sess_id, trials in hmm_looming_decisions.items():\n",
            "    hmm_trial_decisions[sess_id] = {}\n",
            "    for trial_id, looming_decisions in trials.items():\n",
            "        trial_decision = calculate_trial_decision_hmm(looming_decisions)\n",
            "        hmm_trial_decisions[sess_id][trial_id] = trial_decision\n",
            "\n",
            "# Statistics\n",
            "all_hmm_trial_decisions = []\n",
            "for sess_id, trials in hmm_trial_decisions.items():\n",
            "    for trial_id, decision in trials.items():\n",
            "        all_hmm_trial_decisions.append(decision)\n",
            "\n",
            "hmm_decision_counts = Counter(all_hmm_trial_decisions)\n",
            "total = len(all_hmm_trial_decisions)\n",
            "\n",
            "print(f\"HMM Trial Decision Summary (Total: {total} trials):\")\n",
            "print(\"=\"*60)\n",
            "for decision, count in sorted(hmm_decision_counts.items()):\n",
            "    percentage = count / total * 100\n",
            "    print(f\"  {decision}: {count} ({percentage:.1f}%)\")"
        ]
    },
    # Cell 10: Compare with rule-based
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Step 8: Compare HMM decisions with rule-based decisions\n",
            "df_comparison = compare_decisions(deci_trial_dict, hmm_trial_decisions)"
        ]
    },
    # Cell 11: Save results
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Step 9: Save HMM results\n",
            "with open('hmm_looming_decisions.pkl', 'wb') as f:\n",
            "    pickle.dump(hmm_looming_decisions, f)\n",
            "\n",
            "with open('hmm_trial_decisions.pkl', 'wb') as f:\n",
            "    pickle.dump(hmm_trial_decisions, f)\n",
            "\n",
            "with open('hmm_state_mapping.pkl', 'wb') as f:\n",
            "    pickle.dump(state_to_decision, f)\n",
            "\n",
            "print(\"HMM results saved:\")\n",
            "print(\"  ✓ hmm_looming_decisions.pkl\")\n",
            "print(\"  ✓ hmm_trial_decisions.pkl\")\n",
            "print(\"  ✓ hmm_state_mapping.pkl\")\n",
            "print(\"  ✓ hmm_decision_model.pkl\")\n",
            "print(\"\\nHMM decision analysis completed!\")"
        ]
    }
]

# Remove the empty code cell after HMM title
del notebook['cells'][hmm_cell_index + 1]

# Insert new cells
for i, cell in enumerate(new_cells):
    notebook['cells'].insert(insert_position + i, cell)

# Save the modified notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f"\nSuccessfully added {len(new_cells)} new cells to the notebook!")
print("HMM decision analysis cells inserted after the HMM title.")
