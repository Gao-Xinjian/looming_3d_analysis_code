import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os


def plot_lines(series_dict, draw_frames, title_id, lsf=None, xticks=None, save_dir=None):
    """
    Plot multiple line series
    
    Args:
        series_dict: dict of {name: (data, color)}
        draw_frames: (start, end) frame range
        title_id: plot title
        lsf: looming start frame (optional, draws vertical line)
        xticks: custom xticks
        save_dir: directory to save plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    start, end = draw_frames
    
    for label, (data, color) in series_dict.items():
        x = np.arange(start, start + len(data[start:end]))
        ax.plot(x, data[start:end], label=label, color=color, linewidth=2)
    
    if lsf is not None:
        ax.axvline(lsf, color='black', linestyle='--', linewidth=2, label='LSF')
    
    ax.set_xlabel('Frame')
    ax.set_ylabel('Value')
    ax.set_title(title_id)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if xticks is not None:
        ax.set_xticks(xticks)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, f"{title_id}.png")
        fig.savefig(filepath, dpi=150)
    
    plt.show()


def plot_ethogram(bhvr_etho_dict, nest_etho_dict, bhvr_params,
                  draw_frames=None, lsf=None, title='ethogram', save_dir=None):
    """
    Plot ethogram (behavior timeline)
    
    Args:
        bhvr_etho_dict: dict of behavior tuples
        nest_etho_dict: dict of nest presence tuples
        bhvr_params: dict of {behavior: (color, height)}
        draw_frames: (start, end) frame range
        lsf: looming start frame
        title: plot title
        save_dir: directory to save plot
    """
    if draw_frames is None:
        draw_frames = (0, 2000)
    
    start, end = draw_frames
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    trial_idx = 0
    
    for trial_id, bhvr_tuples in bhvr_etho_dict.items():
        for bhvr_name, tuples in bhvr_tuples.items():
            if bhvr_name not in bhvr_params:
                continue
            
            color, height = bhvr_params[bhvr_name]
            
            for t_start, t_end in tuples:
                if t_end > start and t_start < end:
                    t_start = max(t_start, start)
                    t_end = min(t_end, end)
                    
                    rect = Rectangle((t_start, trial_idx), t_end - t_start, height,
                                   facecolor=color, edgecolor='black', linewidth=0.5)
                    ax.add_patch(rect)
        
        if lsf is not None:
            ax.axvline(lsf, color='red', linestyle='--', linewidth=1, alpha=0.5)
        
        trial_idx += 1
    
    ax.set_xlim(start, end)
    ax.set_ylim(0, trial_idx)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Trial')
    ax.set_title(title)
    ax.invert_yaxis()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, f"{title}.png")
        fig.savefig(filepath, dpi=150)
    
    plt.show()
