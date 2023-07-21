from pathlib import Path
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from shutil import rmtree


def split_video(video_file, clip_len_mins=5, vid_len_mins=210):
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


analysis_dir = Path(r"C:\Users\tucke\DLC_Projects\demasoni_singlenuc\analysis")
for video_dir in analysis_dir.glob('*'):
    video_file = video_dir / f'{video_dir.name}.mp4'
    if video_file.exists():
        split_video(video_file)
