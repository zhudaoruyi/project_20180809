import numpy as np
import cv2


class DenseRootSIFT(object):
    def __init__(self):
        self.sift = cv2.xfeatures2d.SIFT_create()

    def detectAndCompute(self, image, step_size=12, window_size=(10, 10)):
        if window_size is None:
            winH, winW = image.shape[:2]
            window_size = (winW // 4, winH // 4)

        descriptors = np.array([], dtype=np.float32).reshape(0, 128)
        kps = np.array([], dtype=np.float32).reshape(0, 2)
        # kps = list()
        for crop in self._crop_image(image, step_size, window_size):
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            kps_, tmp_descriptor = self._detectAndCompute(crop)
            if tmp_descriptor is None:
                continue
            descriptors = np.vstack([descriptors, tmp_descriptor])
            kps_ = np.float32([kp.pt for kp in kps_])
            kps = np.vstack([kps, kps_])
        return kps, descriptors

    def _detect(self, image):
        return self.sift.detect(image)

    def _compute(self, image, kps, eps=1e-7):
        kps, descs = self.sift.compute(image, kps)

        if len(kps) == 0:
            return [], None

        descs /= (descs.sum(axis=1, keepdims=True) + eps)
        descs = np.sqrt(descs)
        return kps, descs

    def _detectAndCompute(self, image):
        kps = self._detect(image)
        return self._compute(image, kps)

    def _sliding_window(self, image, step_size, window_size):
        for y in xrange(0, image.shape[0], step_size):
            for x in xrange(0, image.shape[1], step_size):
                yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

    def _crop_image(self, image, step_size=12, window_size=(10, 10)):
        crops = []
        winH, winW = window_size
        for (x, y, window) in self._sliding_window(image, step_size=step_size, window_size=(winW, winH)):
            if window.shape[0] != winH or window.shape[1] != winW:
                continue
            crops.append(image[y:y+winH, x:x+winW])
        return np.array(crops)
