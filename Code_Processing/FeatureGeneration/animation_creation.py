# ============================================================================
# SKELETON VIDEO RENDERER
# ============================================================================
# Renders low-resolution skeleton videos from world landmarks NPZ file
#
# For each video:
# - White background
# - Black skeleton lines (MediaPipe pose connections)
# - Red dots at landmark positions
# - 3 views side by side: Front, Left Side, Right Side
# - Metadata text in top left corner
#
# Usage:
#   python render_skeleton_videos.py                    # Render all videos
#   python render_skeleton_videos.py --limit 100       # Render first 100 videos
#   python render_skeleton_videos.py --label Move      # Render only Move videos
#   python render_skeleton_videos.py --workers 4       # Use 4 parallel workers
#   python render_skeleton_videos.py --scale 300       # Adjust scale (default 400)
#   python render_skeleton_videos.py --offset-y 0.8    # Adjust vertical position

import os
import json
import argparse
import numpy as np
import cv2
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class RenderConfig:
    # Input
    npz_path: str = "../TrainingDataProcessed/landmarks_world_92_structured_v3.npz"
    metadata_path: str = "../TrainingDataProcessed/dataset_metadata_v3.json"
    
    # Output
    output_dir: str = "../TrainingDataProcessed/rendered_videos"
    
    # Video settings (low res for speed)
    frame_width: int = 200       # Width of each view (total width = 400 * 3 = 1200)
    frame_height: int = 200      # Height (taller to fit upper body)
    fps: int = 25
    codec: str = 'mp4v'
    
    # Rendering
    point_radius: int = 2
    line_thickness: int = 1
    font_scale: float = 0.3
    font_thickness: int = 1
    
    # Colors (BGR)
    bg_color: Tuple[int, int, int] = (255, 255, 255)      # White
    head_color: Tuple[int, int, int] = (0, 150, 255)      # Orange - head/face
    torso_color: Tuple[int, int, int] = (0, 0, 0)         # Black - torso
    arm_color: Tuple[int, int, int] = (255, 0, 0)         # Blue - arms
    hand_color: Tuple[int, int, int] = (0, 0, 255)        # Red - hands
    point_color: Tuple[int, int, int] = (0, 0, 200)       # Dark red - all points
    text_color: Tuple[int, int, int] = (0, 0, 0)          # Black
    low_vis_color: Tuple[int, int, int] = (200, 200, 200) # Gray for low visibility
    
    # Coordinate transform
    scale: float = 150           # Pixels per meter
    offset_y: float = 0.5        # Vertical offset (0.5 = centered)


CONFIG = RenderConfig()


# ============================================================================
# MEDIAPIPE POSE CONNECTIONS (Upper Body Only)
# ============================================================================

# Standard MediaPipe pose connections for upper body (ANATOMICAL)
# Each tuple is (start_landmark_idx, end_landmark_idx)
# Indices: 0=nose, 1-6=eyes, 7-8=ears, 9-10=mouth, 11-12=shoulders, 
#          13-14=elbows, 15-16=wrists, 17-18=pinky, 19-20=index, 21-22=thumb
POSE_CONNECTIONS = [
    # Face - anatomical connections
    (0, 1),                      # Nose to left eye inner
    (1, 2),                      # Left eye inner to left eye
    (2, 3),                      # Left eye to left eye outer
    (3, 7),                      # Left eye outer to left ear
    (0, 4),                      # Nose to right eye inner
    (4, 5),                      # Right eye inner to right eye
    (5, 6),                      # Right eye to right eye outer
    (6, 8),                      # Right eye outer to right ear
    (9, 10),                     # Mouth left to mouth right
    
    # Torso
    (11, 12),                    # Shoulder to shoulder
    
    # Left arm
    (11, 13),                    # Left shoulder to left elbow
    (13, 15),                    # Left elbow to left wrist
    
    # Right arm  
    (12, 14),                    # Right shoulder to right elbow
    (14, 16),                    # Right elbow to right wrist
    
    # Left hand
    (15, 17),                    # Wrist to pinky
    (15, 19),                    # Wrist to index
    (15, 21),                    # Wrist to thumb
    (17, 19),                    # Pinky to index (palm)
    
    # Right hand
    (16, 18),                    # Wrist to pinky
    (16, 20),                    # Wrist to index
    (16, 22),                    # Wrist to thumb
    (18, 20),                    # Pinky to index (palm)
]

# Connection type for coloring: 'head', 'torso', 'arm', 'hand'
CONNECTION_TYPES = {
    # Face
    (0, 1): 'head', (1, 2): 'head', (2, 3): 'head', (3, 7): 'head',
    (0, 4): 'head', (4, 5): 'head', (5, 6): 'head', (6, 8): 'head',
    (9, 10): 'head',
    # Torso
    (11, 12): 'torso',
    # Arms
    (11, 13): 'arm', (13, 15): 'arm',
    (12, 14): 'arm', (14, 16): 'arm',
    # Hands
    (15, 17): 'hand', (15, 19): 'hand', (15, 21): 'hand', (17, 19): 'hand',
    (16, 18): 'hand', (16, 20): 'hand', (16, 22): 'hand', (18, 20): 'hand',
}

# Upper body landmark names (23 landmarks)
LANDMARK_NAMES = [
    'NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_INNER', 
    'RIGHT_EYE', 'RIGHT_EYE_OUTER', 'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 
    'MOUTH_RIGHT', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW',
    'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_INDEX',
    'RIGHT_INDEX', 'LEFT_THUMB', 'RIGHT_THUMB'
]


# ============================================================================
# COORDINATE TRANSFORMATIONS
# ============================================================================

def world_to_screen(x: float, y: float, z: float, 
                   view: str, width: int, height: int,
                   scale: float = 400, 
                   offset_x: float = 0.5, offset_y: float = 0.5) -> Tuple[int, int]:
    """
    Transform world coordinates to screen coordinates for different views.
    
    World coordinate system (MediaPipe):
    - X: Left(-) to Right(+)
    - Y: Up(-) to Down(+)  [negative = above hip, positive = below hip]
    - Z: Towards camera(-) to Away(+)
    
    Note: Coordinates are pre-centered (centroid subtracted) so offset=0.5 centers.
    
    Args:
        x, y, z: World coordinates (centered, in meters)
        view: 'front', 'left', or 'right'
        width, height: Screen dimensions
        scale: Scaling factor (pixels per meter)
        offset_x, offset_y: Offset ratios (0.5 = center)
    
    Returns:
        (screen_x, screen_y) pixel coordinates
    """
    # Scale Z more aggressively for side views (Z is typically small)
    z_scale = scale * 1.0
    
    if view == 'front':
        # Front view: X horizontal, Y vertical
        screen_x = int(width * offset_x + x * scale)
        screen_y = int(height * offset_y + y * scale)
    
    elif view == 'left':
        # Left side view: Z horizontal, Y vertical
        screen_x = int(width * offset_x + z * z_scale)
        screen_y = int(height * offset_y + y * scale)
    
    elif view == 'right':
        # Right side view: Z horizontal (flipped), Y vertical
        screen_x = int(width * offset_x - z * z_scale)
        screen_y = int(height * offset_y + y * scale)
    
    else:
        raise ValueError(f"Unknown view: {view}")
    
    return screen_x, screen_y


def extract_landmark_coords(frame_features: np.ndarray) -> List[Tuple[float, float, float, float]]:
    """
    Extract (x, y, z, visibility) for each of 23 landmarks from 92-feature vector.
    
    Args:
        frame_features: Array of 92 features (23 landmarks × 4 values)
    
    Returns:
        List of 23 tuples: (x, y, z, visibility)
    """
    landmarks = []
    for i in range(23):
        idx = i * 4
        x = frame_features[idx]
        y = frame_features[idx + 1]
        z = frame_features[idx + 2]
        vis = frame_features[idx + 3]
        landmarks.append((x, y, z, vis))
    return landmarks


# ============================================================================
# RENDERING
# ============================================================================

def render_skeleton_frame(landmarks: List[Tuple[float, float, float, float]], 
                         config: RenderConfig) -> np.ndarray:
    """
    Render a single frame with 3 views side by side.
    
    Args:
        landmarks: List of 23 (x, y, z, visibility) tuples
        config: Rendering configuration
    
    Returns:
        Combined frame (height × width*3 × 3)
    """
    w, h = config.frame_width, config.frame_height
    views = ['front', 'left', 'right']
    view_frames = []
    
    # Compute centroid for centering (use visible landmarks only)
    xs = [lm[0] for lm in landmarks if lm[3] > 0.3]
    ys = [lm[1] for lm in landmarks if lm[3] > 0.3]
    zs = [lm[2] for lm in landmarks if lm[3] > 0.3]
    
    center_x = np.mean(xs) if xs else 0
    center_y = np.mean(ys) if ys else 0
    center_z = np.mean(zs) if zs else 0
    
    for view in views:
        # Create white background
        frame = np.full((h, w, 3), config.bg_color, dtype=np.uint8)
        
        # Convert world coords to screen coords for this view
        # Center the skeleton by subtracting centroid
        screen_points = []
        for lm in landmarks:
            x, y, z, vis = lm
            
            # Center coordinates
            x_centered = x - center_x
            y_centered = y - center_y
            z_centered = z - center_z
            
            sx, sy = world_to_screen(x_centered, y_centered, z_centered, view, w, h, 
                                    scale=config.scale, offset_y=config.offset_y)
            screen_points.append((sx, sy, vis))
        
        # Draw skeleton lines with color by body part
        for start_idx, end_idx in POSE_CONNECTIONS:
            if start_idx >= len(screen_points) or end_idx >= len(screen_points):
                continue
            
            sx1, sy1, vis1 = screen_points[start_idx]
            sx2, sy2, vis2 = screen_points[end_idx]
            
            # Allow points slightly outside frame (don't skip too aggressively)
            margin = 50
            if not (-margin <= sx1 < w+margin and -margin <= sy1 < h+margin and 
                    -margin <= sx2 < w+margin and -margin <= sy2 < h+margin):
                continue
            
            # Get color based on connection type
            avg_vis = (vis1 + vis2) / 2
            if avg_vis < 0.5:
                color = config.low_vis_color
            else:
                conn_type = CONNECTION_TYPES.get((start_idx, end_idx), 'torso')
                if conn_type == 'head':
                    color = config.head_color
                elif conn_type == 'arm':
                    color = config.arm_color
                elif conn_type == 'hand':
                    color = config.hand_color
                else:
                    color = config.torso_color
            
            cv2.line(frame, (sx1, sy1), (sx2, sy2), color, config.line_thickness)
        
        # Draw landmark points
        for i, (sx, sy, vis) in enumerate(screen_points):
            # Allow points slightly outside frame
            margin = 20
            if not (-margin <= sx < w+margin and -margin <= sy < h+margin):
                continue
            
            # Clamp to frame boundaries for drawing
            draw_x = max(0, min(w-1, sx))
            draw_y = max(0, min(h-1, sy))
            
            # Use gray for low visibility points
            if vis < 0.5:
                color = config.low_vis_color
            else:
                color = config.point_color
            
            cv2.circle(frame, (draw_x, draw_y), config.point_radius, color, -1)
        
        # Add view label
        cv2.putText(frame, view.upper(), (10, h - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, config.font_scale, 
                   config.text_color, config.font_thickness)
        
        view_frames.append(frame)
    
    # Combine views horizontally
    combined = np.hstack(view_frames)
    
    # Add vertical divider lines between views
    h = config.frame_height
    w = config.frame_width
    cv2.line(combined, (w, 0), (w, h), (150, 150, 150), 2)      # After FRONT
    cv2.line(combined, (w*2, 0), (w*2, h), (150, 150, 150), 2)  # After LEFT
    
    return combined


def render_video(video_data: Dict, config: RenderConfig) -> Optional[str]:
    """
    Render a single video from world landmarks.
    
    Args:
        video_data: Dict with 'landmarks', 'metadata', 'timestamps'
        config: Rendering configuration
    
    Returns:
        Output filepath or None if failed
    """
    metadata = video_data['metadata']
    landmarks_array = video_data['landmarks']
    
    if len(landmarks_array) == 0:
        return None
    
    # Existing code for paths
    base_name = metadata['filename'].replace('.mp4', '_skeleton.mp4')
    label_dir = os.path.join(config.output_dir, metadata['training_label'])
    os.makedirs(label_dir, exist_ok=True)
    output_path = os.path.join(label_dir, base_name)

    # --- ADD THIS CODE START ---
    # Create the path for the small NPZ (e.g., video_name_skeleton.npz)
    npz_output_path = output_path.replace('.mp4', '.npz')
    
    # Save the individual landmark data and its metadata
    np.savez_compressed(
        npz_output_path, 
        landmarks=landmarks_array,
        metadata=metadata)
    
    # Create output filename
    base_name = metadata['filename'].replace('.mp4', '_skeleton.mp4')
    label_dir = os.path.join(config.output_dir, metadata['training_label'])
    os.makedirs(label_dir, exist_ok=True)
    output_path = os.path.join(label_dir, base_name)
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*config.codec)
    total_width = config.frame_width * 3
    writer = cv2.VideoWriter(output_path, fourcc, config.fps, 
                            (total_width, config.frame_height))
    
    if not writer.isOpened():
        print(f"Error: Could not create video writer for {output_path}")
        return None
    
    # Metadata text
    meta_lines = [
        f"Corpus: {metadata['corpus']}",
        f"Speaker: {metadata['speaker']}",
        f"Category: {metadata['training_label']}",
        f"Subtype: {metadata['subtype']}",
        f"Clip: {metadata['clip_id']}",
        f"Mirror: {'Yes' if metadata['is_mirror'] else 'No'}"
    ]
    
    # Render each frame
    for frame_idx, frame_features in enumerate(landmarks_array):
        landmarks = extract_landmark_coords(frame_features)
        frame = render_skeleton_frame(landmarks, config)
        
        # Add metadata text to top left
        y_offset = 15
        for line in meta_lines:
            cv2.putText(frame, line, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, config.font_scale,
                       config.text_color, config.font_thickness)
            y_offset += 15
        
        # Add frame counter
        cv2.putText(frame, f"Frame: {frame_idx + 1}/{len(landmarks_array)}", 
                   (10, y_offset + 5), cv2.FONT_HERSHEY_SIMPLEX, 
                   config.font_scale, config.text_color, config.font_thickness)
        
        writer.write(frame)
    
    writer.release()
    return output_path


def render_video_wrapper(args: Tuple[Dict, RenderConfig]) -> Optional[str]:
    """Wrapper for multiprocessing."""
    video_data, config = args
    try:
        return render_video(video_data, config)
    except Exception as e:
        print(f"Error rendering {video_data['metadata']['filename']}: {e}")
        return None


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Render skeleton videos from world landmarks')
    parser.add_argument('--npz', type=str, default=CONFIG.npz_path,
                       help='Path to world landmarks NPZ file')
    parser.add_argument('--output', type=str, default=CONFIG.output_dir,
                       help='Output directory for rendered videos')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of videos to render')
    parser.add_argument('--label', type=str, default=None,
                       help='Only render videos with this label (NoGesture/Gesture/Move/Objman)')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers')
    parser.add_argument('--fps', type=int, default=CONFIG.fps,
                       help='Output video FPS')
    parser.add_argument('--width', type=int, default=RenderConfig.frame_width) 
    parser.add_argument('--height', type=int, default=RenderConfig.frame_height) 
    parser.add_argument('--scale', type=float, default=RenderConfig.scale)
    parser.add_argument('--offset-y', type=float, default=RenderConfig.offset_y) 
     
    
    args = parser.parse_args()
    
    # Update config
    config = RenderConfig()
    config.npz_path = args.npz
    config.output_dir = args.output
    config.frame_width = args.width
    config.frame_height = args.height
    config.fps = args.fps
    config.scale = args.scale
    config.offset_y = args.offset_y
    
    num_workers = args.workers or max(1, cpu_count() - 2)
    
    print("\n" + "="*70)
    print("SKELETON VIDEO RENDERER")
    print("="*70)
    print(f"Input NPZ: {config.npz_path}")
    print(f"Output dir: {config.output_dir}")
    print(f"Resolution: {config.frame_width * 3} × {config.frame_height}")
    print(f"FPS: {config.fps}")
    print(f"Scale: {config.scale} px/m, Offset Y: {config.offset_y}")
    print(f"Workers: {num_workers}")
    if args.limit:
        print(f"Limit: {args.limit} videos")
    if args.label:
        print(f"Label filter: {args.label}")
    print("="*70 + "\n")
    
    # Load NPZ
    print("Loading world landmarks...")
    if not os.path.exists(config.npz_path):
        print(f"ERROR: NPZ file not found: {config.npz_path}")
        return
    
    data = np.load(config.npz_path, allow_pickle=True)
    
    # Collect videos to render
    labels = ['NoGesture', 'Gesture', 'Move', 'Objman']
    if args.label:
        if args.label not in labels:
            print(f"ERROR: Invalid label '{args.label}'. Must be one of {labels}")
            return
        labels = [args.label]
    
    videos_to_render = []
    
    for label in labels:
        landmarks_key = f'{label}_landmarks'
        metadata_key = f'{label}_metadata'
        
        if landmarks_key not in data:
            print(f"Warning: No data for label '{label}'")
            continue
        
        landmarks_array = data[landmarks_key]
        metadata_array = data[metadata_key]
        
        print(f"  {label}: {len(landmarks_array)} videos")
        
        for i in range(len(landmarks_array)):
            video_data = {
                'landmarks': landmarks_array[i],
                'metadata': metadata_array[i]
            }
            videos_to_render.append(video_data)
    
    print(f"\nTotal videos to render: {len(videos_to_render)}")
    
    # Apply limit
    if args.limit and args.limit < len(videos_to_render):
        videos_to_render = videos_to_render[:args.limit]
        print(f"Limited to: {len(videos_to_render)} videos")
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Render videos
    print("\nRendering videos...")
    
    args_list = [(v, config) for v in videos_to_render]
    
    successful = 0
    failed = 0
    
    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(render_video_wrapper, args_list),
            total=len(args_list),
            desc="Rendering"
        ))
    
    for r in results:
        if r is not None:
            successful += 1
        else:
            failed += 1
    
    print("\n" + "="*70)
    print("RENDERING COMPLETE")
    print("="*70)
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output directory: {config.output_dir}")
    
    # List subdirectories
    for label in labels:
        label_dir = os.path.join(config.output_dir, label)
        if os.path.exists(label_dir):
            count = len([f for f in os.listdir(label_dir) if f.endswith('.mp4')])
            print(f"  {label}/: {count} videos")
    
    print("\n")


if __name__ == "__main__":
    main()