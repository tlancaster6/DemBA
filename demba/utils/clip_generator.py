"""Video clip generation for tracklet annotation."""

import cv2
import logging
import subprocess
from pathlib import Path

from demba import config
from demba.identity_correction import PatchExtractor
from demba.utils.dlc import load_tracklets

logger = logging.getLogger(__name__)


def generate_tracklet_clip(
    trial_manager,
    tracklet_idx,
    clip_id,
    output_path,
    bbox_color=None,
    bbox_thickness=None,
    text_fontsize=None,
    text_thickness=None,
    text_bg_color=(0, 0, 0),
    text_fg_color=(255, 255, 255)
):
    """
    Render a video clip for a single tracklet with bounding box overlay.

    Parameters
    ----------
    trial_manager : TrialManager
        Trial manager instance
    tracklet_idx : int
        Tracklet to render
    clip_id : str
        Unique clip identifier (e.g., 'clip_042')
    output_path : Path
        Output MP4 file path
    bbox_color : tuple, optional
        BGR color for bounding box. Defaults to config.DEFAULT_EVAL_BBOX_COLOR
    bbox_thickness : int, optional
        Thickness of bounding box border. Defaults to config.DEFAULT_EVAL_BBOX_THICKNESS
    text_fontsize : float, optional
        Font scale for overlay text. Defaults to config.DEFAULT_EVAL_TEXT_FONTSIZE
    text_thickness : int, optional
        Thickness for text. Defaults to config.DEFAULT_EVAL_TEXT_THICKNESS
    text_bg_color : tuple
        Background color for text box (BGR)
    text_fg_color : tuple
        Foreground color for text (BGR)

    Returns
    -------
    dict : Clip metadata
        {
            'clip_id': str,
            'video_name': str,
            'tracklet_id': int,
            'start_frame': int,
            'end_frame': int,
            'duration_sec': float,
            'n_frames': int
        }

    Implementation
    --------------
    1. Load tracklet from pickle
    2. Get frame numbers for this tracklet
    3. Open video file
    4. For each frame in tracklet:
       a. Read frame from video
       b. Load keypoints for this frame
       c. Calculate bounding box from keypoints (using PatchExtractor)
       d. Draw bounding box on frame
       e. Add text overlay (clip_id, video_name) at top
    5. Write frames to output MP4
    6. Return metadata
    """
    # Set defaults
    if bbox_color is None:
        bbox_color = config.DEFAULT_EVAL_BBOX_COLOR
    if bbox_thickness is None:
        bbox_thickness = config.DEFAULT_EVAL_BBOX_THICKNESS
    if text_fontsize is None:
        text_fontsize = config.DEFAULT_EVAL_TEXT_FONTSIZE
    if text_thickness is None:
        text_thickness = config.DEFAULT_EVAL_TEXT_THICKNESS

    # Load tracklets
    tracklets, _ = load_tracklets(trial_manager.el_pickle_path())
    tracklet = tracklets[tracklet_idx]

    # Initialize patch extractor for bbox calculation
    patch_extractor = PatchExtractor(
        video_path=trial_manager.video_path(),
        padding=config.DEFAULT_EVAL_BBOX_PADDING
    )

    # Open video
    video_path = trial_manager.video_path()
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = config.VIDEO_FPS  # Fallback to config
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize video writer
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use mp4v for initial creation (will re-encode with H.264 after)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not out.isOpened():
        cap.release()
        patch_extractor.close_video()
        raise RuntimeError(f"Failed to initialize VideoWriter for {output_path}")

    # Process each frame in tracklet
    frame_indices = tracklet.inds
    start_frame = frame_indices[0]
    end_frame = frame_indices[-1]
    n_frames_written = 0

    for local_idx, absolute_frame in enumerate(frame_indices):
        # Seek to frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, absolute_frame)
        ret, frame = cap.read()

        if not ret:
            logger.warning(f"Could not read frame {absolute_frame}")
            continue

        # Get keypoints for this frame (shape: n_bodyparts x 3)
        keypoints = tracklet.data[local_idx, :, :3]

        # Calculate and draw bounding box
        bbox = patch_extractor.compute_bbox(keypoints)
        if bbox is not None:
            min_x, min_y, max_x, max_y = bbox
            cv2.rectangle(frame,
                         (int(min_x), int(min_y)),
                         (int(max_x), int(max_y)),
                         color=bbox_color,
                         thickness=bbox_thickness)
        else:
            logger.debug(f"No valid bbox for tracklet {tracklet_idx} frame {absolute_frame}")

        # Write frame
        out.write(frame)
        n_frames_written += 1

    # Cleanup
    cap.release()
    out.release()
    patch_extractor.close_video()

    # Verify clip was created
    if not output_path.exists():
        raise RuntimeError(f"Failed to create clip: {output_path}")

    if n_frames_written == 0:
        logger.warning(f"No frames written for clip {clip_id}")

    # Re-encode with H.264 for better compatibility with media players
    # This ensures clips can be opened in VLC, Windows Media Player, etc.
    temp_path = output_path.with_suffix('.tmp.mp4')
    try:
        subprocess.run(
            [
                'ffmpeg', '-i', str(output_path),
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                '-pix_fmt', 'yuv420p',
                '-y',  # Overwrite
                str(temp_path)
            ],
            check=True,
            capture_output=True,
            text=True
        )
        # Replace original with re-encoded version
        temp_path.replace(output_path)
        logger.debug(f"Re-encoded {clip_id} with H.264")
    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to re-encode {clip_id} with H.264: {e.stderr}")
        # Keep the original mp4v version if re-encoding fails
        if temp_path.exists():
            temp_path.unlink()
    except FileNotFoundError:
        logger.warning(f"ffmpeg not found, using mp4v codec (may not play in all players)")

    # Return metadata
    duration_sec = len(frame_indices) / fps

    return {
        'clip_id': clip_id,
        'video_name': video_path.name,
        'tracklet_id': tracklet_idx,
        'start_frame': int(start_frame),
        'end_frame': int(end_frame),
        'duration_sec': float(duration_sec),
        'n_frames': len(frame_indices)
    }