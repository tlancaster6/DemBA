from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from skimage.color import rgb2hsv
from skimage.morphology import remove_small_holes, remove_small_objects
import numpy as np
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks


def plot_roi_over_img(img_path, cx, cy, r):
    """draws a red circle centered at (cx, cy) with radius (r) over the image"""
    img = plt.imread(img_path)
    fig, ax = plt.subplots(1)
    circ = Circle((cx, cy), r, edgecolor='g', linewidth=5)
    ax.imshow(img)
    ax.add_patch(circ)
    return fig


def estimate_roi(img_path, rmin=125, rmax=250, hmin=0.4, hmax=0.5):
    """uses a combination of hue thresholding, morphological manipulations, and a hough circle transform to
    automatically estimate the coordinates and radius (in pixels) of a circle that encloses the roi"""
    img = rgb2hsv(plt.imread(img_path))  # filtering for a particular color is easier in hsv than rgb
    mask = np.where((hmin < img[:, :, 0]) & (img[:, :, 0] < hmax), 1, 0)  # mask to the defined hue range
    mask = mask.astype(bool)
    mask = remove_small_holes(mask, 60000)  # close the interior of the pipe
    mask = remove_small_objects(mask, 1000)  # remove small disconnected patches of the desired color
    edges = canny(mask, 3)  # find the edges of the pipe
    hough_radii = np.arange(rmin, rmax, 5)  # provide a range possible radii for the roi circle
    result = hough_circle(edges, hough_radii)  # calculate the hough transform
    _, cx, cy, r = hough_circle_peaks(result, hough_radii, total_num_peaks=1)  # isolate the most likely candidate
    cx, cy, r = [x[0] for x in [cx, cy, r]] # flatten the results
    return cx, cy, r


