from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from skimage.color import rgb2hsv, rgb2gray
from skimage.morphology import remove_small_holes, remove_small_objects
import numpy as np
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks
import cv2
import pathlib

def generate_roi_visualization(img, cx, cy, r, output_path):
    """draws a red circle centered at (cx, cy) with radius (r) over the image"""
    fig, ax = plt.subplots(1)
    img = rgb2gray(img)
    circ = Circle((cx, cy), r, edgecolor='g', linewidth=5, fill=False)
    ax.imshow(img, cmap='gray')
    ax.add_patch(circ)
    fig.savefig(output_path)
    plt.close(fig)

def estimate_roi(frame, output_path=None):
    frame_height, frame_width = frame.shape[:-1]
    roi_x = frame_width / 2
    roi_y = frame_height / 2
    roi_r = min(frame_height, frame_width) / 2
    if output_path is not None:
        generate_roi_visualization(frame, roi_x, roi_y, roi_r, output_path)
    return roi_x, roi_y, roi_r, frame_height, frame_width



def estimate_roi_hough(img, rmin=125, rmax=250, hmin=0.4, hmax=0.5, output_path=None):
    """uses a combination of hue thresholding, morphological manipulations, and a hough circle transform to
    automatically estimate the coordinates and radius (in pixels) of a circle that encloses the roi"""
    img_rgb = img.copy()
    img = rgb2hsv(img)  # filtering for a particular color is easier in hsv than rgb
    mask = np.where((hmin < img[:, :, 0]) & (img[:, :, 0] < hmax), 1, 0)  # mask to the defined hue range
    mask = mask.astype(bool)
    mask = remove_small_holes(mask, 60000)  # close the interior of the pipe
    mask = remove_small_objects(mask, 1000)  # remove small disconnected patches of the desired color
    edges = canny(mask, 3)  # find the edges of the pipe
    hough_radii = np.arange(rmin, rmax, 5)  # provide a range possible radii for the roi circle
    result = hough_circle(edges, hough_radii)  # calculate the hough transform
    _, cx, cy, r = hough_circle_peaks(result, hough_radii, total_num_peaks=1)  # isolate the most likely candidate
    cx, cy, r = [x[0] for x in [cx, cy, r]]  # flatten the results
    if output_path:
        generate_roi_visualization(img_rgb, cx, cy, r, output_path)
    return cx, cy, r

def crop_video_to_roi(video_path):
    input_path = pathlib.Path(video_path)
    output_path = input_path.with_name(input_path.stem + '_cropped' + input_path.suffix)
    vis_out = input_path.with_name(input_path.stem + '_roi.jpg')
    cap = cv2.VideoCapture(str(input_path))
    ret, frame = cap.read()
    cx, cy, r = estimate_roi(frame, output_path=vis_out)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (2*r, 2*r))
    while True:
        cropped_frame = frame[cy-r:cy+r, cx-r:cx+r]
        writer.write(cropped_frame)
        ret, frame = cap.read()
        if not ret:
            break
    cap.release()
    writer.release()
