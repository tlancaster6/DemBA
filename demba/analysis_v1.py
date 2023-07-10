from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from skimage.color import rgb2hsv, rgb2gray
from skimage.morphology import remove_small_holes, remove_small_objects
import numpy as np
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks
import cv2
import ruptures as rpt
import pandas as pd
from itertools import permutations
from pathlib import Path
idx = pd.IndexSlice


def threshed_pelt_event_detection(data: pd.Series, upper_thresh: float, min_size: int, penalty=3):
    binary_data = (data < upper_thresh).astype(int).values
    bkpts = rpt.Pelt(model='l2', min_size=min_size, jump=1).fit_predict(binary_data, penalty)
    if bkpts[0] != 0:
        bkpts = [0] + bkpts
    if bkpts[-1] != len(data):
        bkpts = bkpts + [len(data)]
    event_ids = pd.Series(-1, index=data.index)
    current_event = 0
    for start, stop in list(zip(bkpts[:-1], bkpts[1:])):
        if np.mean(binary_data[start:stop]) > 0.5:
            event_ids.iloc[start:stop] = current_event
            current_event += 1
    return event_ids


class AnalyzerV1:

    def __init__(self, video_path):
        self.video_path = video_path
        self.h5_path = str(next(Path(self.video_path).parent.glob(Path(self.video_path).stem + '*_filtered.h5')))
        self.roi_x, self.roi_y, self.roi_r = self.estimate_roi()
        self.frame_h, self.frame_w = self.get_frame_size()
        self.pose_df = self.load_pose_data()
        self.individuals, self.bodyparts = [list(self.pose_df.columns.levels[i]) for i in [0, 1]]
    def load_pose_data(self):
        pose_df = pd.read_hdf(self.h5_path)
        scorer = pose_df.columns.get_level_values(0)[0]
        pose_df = pose_df.loc[:, scorer]
        pose_df = pose_df.loc[:, idx[:, :, ('x', 'y')]]
        pose_df.columns = pose_df.columns.remove_unused_levels()
        pose_df = pose_df.sort_index(axis=1)

    def estimate_roi(self, rmin=125, rmax=250, hmin=0.4, hmax=0.5, visualize=True):
        """uses a combination of hue thresholding, morphological manipulations, and a hough circle transform to
        automatically estimate the coordinates and radius (in pixels) of a circle that encloses the roi"""
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_FRAME_COUNT)//2)
        ret, frame = cap.read()
        cap.release()
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = rgb2hsv(img_rgb)  # filtering for a particular color is easier in hsv than rgb
        mask = np.where((hmin < img[:, :, 0]) & (img[:, :, 0] < hmax), 1, 0)  # mask to the defined hue range
        mask = mask.astype(bool)
        mask = remove_small_holes(mask, 60000)  # close the interior of the pipe
        mask = remove_small_objects(mask, 1000)  # remove small disconnected patches of the desired color
        edges = canny(mask, 3)  # find the edges of the pipe
        hough_radii = np.arange(rmin, rmax, 5)  # provide a range possible radii for the roi circle
        result = hough_circle(edges, hough_radii)  # calculate the hough transform
        _, cx, cy, r = hough_circle_peaks(result, hough_radii, total_num_peaks=1)  # isolate the most likely candidate
        cx, cy, r = [x[0] for x in [cx, cy, r]]  # flatten the results

        if visualize:
            roi_vis_path = str(Path(self.video_path).parent / (Path(self.video_path).stem + '_roi.png'))
            fig, ax = plt.subplots(1)
            img = rgb2gray(img)
            circ = Circle((cx, cy), r, edgecolor='g', linewidth=5, fill=False)
            ax.imshow(img, cmap='gray')
            ax.add_patch(circ)
            fig.savefig(roi_vis_path)
            plt.close(fig)

        return cx, cy, r

    def get_frame_size(self):
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        cap.release()
        h, w = frame.shape[:-1]
        return h, w

    def detect_mouthing_events(self, dist_thresh=25, min_frames=30):
        candidate_dists = []
        for id1, id2 in list(permutations(self.individuals, 2)):
            dists = self.pose_df.loc[:, idx[id1, 'nose', :]].values - self.pose_df.loc[:, idx[id2, 'stripe4', :]].values
            candidate_dists.append(np.hypot(dists[:, 0], dists[:, 1]))
        min_dists = pd.Series(np.nanmin(np.vstack(candidate_dists), axis=0), name='min_dist_nose_to_stripe4')
        event_ids = threshed_pelt_event_detection(min_dists, dist_thresh, min_frames)

