# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>

"""
import os
import numpy as np
import cv2


def load(filepath, mode=None):
    if not os.path.exists(filepath):
        raise IOError("No such file or directory: '{}'".format(filepath))

    if mode is None:
        mode = cv2.IMREAD_UNCHANGED
    img = cv2.imread(filepath, mode)

    if img.ndim > 2:
        if img.shape[-1] == 4:
            color_mode = cv2.COLOR_BGRA2RGBA
        else:
            color_mode = cv2.COLOR_BGR2RGB

        img = cv2.cvtColor(img, color_mode)
    return img


def save(filepath, img):
    img = np.asanyarray(img)

    if img.ndim == 2:
        cv2.imwrite(filepath, img)
    else:
        if img.shape[-1] == 4:
            color_mode = cv2.COLOR_RGBA2BGRA
        else:
            color_mode = cv2.COLOR_RGB2BGR
        if not cv2.imwrite(filepath, cv2.cvtColor(img, color_mode)):
            dirname = os.path.dirname(filepath)
            if not os.path.exists(dirname):
                msg = "No such directory: '{}'".format(dirname)
            else:
                msg = "Cannot write image to '{}'".format(filepath)
            raise IOError(msg)
