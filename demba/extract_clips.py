from pathlib import Path
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from shutil import rmtree
import cv2
import numpy as np

def split_video_ffmpeg(video_file, clip_len_mins=1, vid_len_mins=210):
    clip_len_secs = int(clip_len_mins * 60)
    t0 = 0
    t1 = clip_len_secs
    output_dir = Path(video_file).parent / 'clips'
    if output_dir.exists():
        rmtree(str(output_dir))
    output_dir.mkdir()
    while t1 <= (vid_len_mins * 60):
        output_file = output_dir / f'{Path(video_file).stem}_{t0//60}-{t1//60}.mp4'
        ffmpeg_extract_subclip(str(video_file), t0, t1, str(output_file))
        t0 += clip_len_secs
        t1 += clip_len_secs


def split_video_opencv(video_file, clip_len_frames=1800, overwrite=True):
    print(f'splitting {Path(video_file).name}')
    output_dir = Path(video_file).parent / 'clips'
    if output_dir.exists():
        if overwrite:
            rmtree(str(output_dir))
        else:
            print(f'splitting already complete for {Path(video_file).name}')
            return
    output_dir.mkdir()
    cap = cv2.VideoCapture(str(video_file))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    splits = [int(i) for i in np.arange(0, n_frames, clip_len_frames).tolist()]
    splits = list(zip(splits[:-1], splits[1:]))
    for f0, f1 in splits:
        output_file = str(output_dir / f'{Path(video_file).stem}_{f0:06}-{f1-1:06}.mp4')
        writer = cv2.VideoWriter(output_file, fourcc, fps, size)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
        ret = True
        while ret:
            ret, frame = cap.read()
            frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
            if frame_number < f1:
                writer.write(frame)
            else:
                break
        writer.release()
    cap.release()





analysis_dir = Path(r"C:\Users\tucke\DLC_Projects\demasoni_singlenuc\analysis")
for video_dir in analysis_dir.glob('*'):
    video_file = video_dir / f'{video_dir.name}.mp4'
    if video_file.exists():
        split_video_opencv(video_file, overwrite=False)