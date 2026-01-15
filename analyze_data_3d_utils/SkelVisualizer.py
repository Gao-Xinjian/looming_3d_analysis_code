import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def draw_mean_skeleton(coords, joint_names, skel_conns):
    """
    Draw mean skeleton from coordinate data
    
    Args:
        coords: array of shape (n_frames, n_joints, 3)
        joint_names: list of joint names
        skel_conns: list of (joint1_idx, joint2_idx) skeleton connections
    """
    # Compute mean position
    mean_coords = np.mean(coords, axis=0)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot joints
    ax.scatter(mean_coords[:, 0], mean_coords[:, 1], mean_coords[:, 2],
              c='red', s=50, alpha=0.8)
    
    # Plot skeleton connections
    for conn_idx, (j1, j2) in enumerate(skel_conns):
        if j1 < len(mean_coords) and j2 < len(mean_coords):
            pts = mean_coords[[j1, j2], :]
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], 'b-', linewidth=2)
    
    # Labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Mean Skeleton')
    
    plt.show()


def visualize_3d(coords, frame_range, joint_names, skel_conns, c_maps, 
                video_path=None, bhvr_labels=None, mode='global'):
    """
    Visualize 3D skeleton
    
    Args:
        coords: array of shape (n_frames, n_joints, 3)
        frame_range: (start, end, step)
        joint_names: list of joint names
        skel_conns: skeleton connections
        c_maps: color maps
        video_path: optional video path
        bhvr_labels: optional behavior labels
        mode: 'global' or 'relative'
    """
    start, end, step = frame_range
    
    for frame_idx in range(start, end, step):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        frame_coords = coords[frame_idx]
        
        # Plot joints
        ax.scatter(frame_coords[:, 0], frame_coords[:, 1], frame_coords[:, 2],
                  c='red', s=100, alpha=0.8)
        
        # Plot connections
        for j1, j2 in skel_conns:
            if j1 < len(frame_coords) and j2 < len(frame_coords):
                pts = frame_coords[[j1, j2], :]
                ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], 'b-', linewidth=2)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        if bhvr_labels is not None:
            ax.set_title(f'Frame {frame_idx}: {bhvr_labels[frame_idx]}')
        else:
            ax.set_title(f'Frame {frame_idx}')
        
        plt.show()


def visualize_and_save_3d(coords, frame_range, joint_names, skel_conns, c_maps,
                         save_path, viz_fps, video_path=None, bhvr_labels=None, mode='global'):
    """
    Visualize and save 3D skeleton as video
    
    Args:
        coords: array of shape (n_frames, n_joints, 3)
        frame_range: (start, end, step)
        joint_names: list of joint names
        skel_conns: skeleton connections
        c_maps: color maps
        save_path: path to save video
        viz_fps: visualization fps
        video_path: optional original video path
        bhvr_labels: optional behavior labels
        mode: 'global' or 'relative'
    """
    try:
        import cv2
    except ImportError:
        print("OpenCV not installed. Skipping video save.")
        return
    
    start, end, step = frame_range
    
    # Determine output size
    frame_width, frame_height = 800, 600
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, viz_fps, (frame_width, frame_height))
    
    print(f"Saving visualization to {save_path}")
    
    for frame_idx in range(start, end, step):
        # This is a simplified version - actual implementation would render to image
        frame_coords = coords[frame_idx]
        
        # Create a simple visualization frame
        frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        
        # Draw skeleton
        # (simplified - actual implementation would use proper 3D projection)
        
        out.write(frame)
    
    out.release()
    print("Done!")
