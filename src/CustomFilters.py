import math

import cv2
import numpy as np


class WLSFilter:
    def __init__(self, _lambda, _sigma, fov, baseline):

        self._lambda = _lambda
        self._sigma = _sigma

        self.fov = fov
        self.baseline = baseline

        self.wlsFilter = cv2.ximgproc.createDisparityWLSFilterGeneric(False)

    def filter(self, disparity_frame, right):
        focal = disparity_frame.shape[1] / (2. * math.tan(math.radians(self.fov / 2)))
        depth_scale_factor = self.baseline * focal

        print(depth_scale_factor)

        self.wlsFilter.setLambda(self._lambda)
        self.wlsFilter.setSigmaColor(self._sigma)

        filtered_disp = self.wlsFilter.filter(disparity_frame, right)

        with np.errstate(divide='ignore'):  # Should be safe to ignore div by zero here
            # raw depth values
            depth_frame = (depth_scale_factor / filtered_disp).astype(np.uint16)

        return depth_frame
