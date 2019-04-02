# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>

"""
import os
from functools import wraps
import re
import warnings

from . import img_utils
from .io_utils import get_files_by_extension
from .io_utils import create_directory_if_not_exists
from .io_utils import read_json


# Sets
TRAIN_SET = 'training'
VALID_SET = 'validation'
TEST_SET = 'test'
SETS = (TRAIN_SET, VALID_SET, TEST_SET)
TRAIN_PERSON_IDS = (3, 4, 20, 6, 39, 34, 37, 9, 27, 29, 18, 5, 10, 17, 16, 24,
                    30, 32, 35, 33, 40)
VALID_PERSON_IDS = (26, 15, 7, 2, 8, 12, 28)
TEST_PERSON_IDS = (19, 14, 13, 38, 36, 11, 0, 31, 1)

# Sizes
SIZE_SMALL = 'small'
SIZE_LARGE = 'large'
SIZES = (SIZE_SMALL, SIZE_LARGE)
DEFAULT_SIZE = SIZE_SMALL

# Subfolders
_PATCH_SUFFIX = '_patches'
_JSON_SUBFOLDER = 'json'

# Define basename suffixes
_DEPTH_SUFFIX = '_Depth.pgm'
_MASK_SUFFIX = '_Mask.png'
_RGB_SUFFIX = '_RGB.png'
_PCD_SUFFIX = '_cloud.pcd'
_JSON_SUFFIX = '.json'


class RGBDOrientationDataset(object):
    """
    Dataset container class.

    Parameters
    ----------
    dataset_basepath : str
        Path to dataset root, e.g. '/dataset/orientation/'.
    set_name : str
        Set to load, should be one of 'training', 'validation' or 'test'.
    default_size : str
        Default image size to use. Should be either 'small' or 'large',
        default: 'small'.
    """
    def __init__(self, dataset_basepath, set_name, default_size):
        assert set_name in SETS

        # store arguments
        self._dataset_basepath = dataset_basepath
        self._set_name = set_name
        self._default_size = default_size

        # get all json files
        json_path = os.path.join(self._dataset_basepath, self._set_name,
                                 _JSON_SUBFOLDER)
        json_files = get_files_by_extension(json_path,
                                            extension='.json',
                                            flat_structure=True,
                                            recursive=True,
                                            follow_links=True)

        # load samples
        self._samples = [Sample.from_filepath(fp, self._default_size)
                         for fp in json_files]

    @property
    def dataset_basepath(self):
        return self._dataset_basepath

    @property
    def set_name(self):
        return self._set_name

    @property
    def default_size(self):
        return self._default_size

    def extract_all_patches(self, size=None, with_progressbar=True):
        """
        Extract all patches at once.

        Parameters
        ----------
        size : str
            Image size to extract the patches for. Should be either 'small' or
            'large'. If `size` is None, the default size of the dataset object
            is used.
        with_progressbar : bool
            If `with_progressbar` is True, `tqdm` is used to display a progress
            bar.

        """
        assert size in (None, ) + SIZES
        size = size or self._default_size

        # small helper function for tqdm
        def _tqdm_wrapper(iterable, **tqdm_kwargs):
            if with_progressbar:
                from tqdm import tqdm
                return tqdm(iterable, unit="images", **tqdm_kwargs)
            else:
                return iterable

        # extract patches
        for s in _tqdm_wrapper(self._samples):
            # define images to process and corresponding filepaths
            imgs = [s.get_depth_img(size=size), s.get_mask_img(size=size),
                    s.get_rgb_img(size=size)]
            filepaths = [s.get_depth_patch_filepath(size=size),
                         s.get_mask_patch_filepath(size=size),
                         s.get_rgb_patch_filepath(size=size)]
            for img, filepath in zip(imgs, filepaths):
                # extract patch
                patch = Sample.extract_patch(img=img,
                                             roi_y=s.get_roi_y(),
                                             roi_x=s.get_roi_x())
                # create directory structure if not exists
                create_directory_if_not_exists(os.path.dirname(filepath))
                # save patch
                img_utils.save(filepath, patch)

    def strip_to_multiple_of_batch_size(self, batch_size):
        """
        Strip samples to multiple of the given batch size.

        Parameters
        ----------
        batch_size : int
            The batch size.

        """
        n_batches = len(self._samples) // batch_size
        self._samples = self._samples[:n_batches*batch_size]

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, index):
        return self._samples[index]


class Sample(object):
    """
    Simple container for a single dataset sample.

    Parameters
    ----------
    basename : str
        Sample basename, e.g. 'p4-plain-Kinect1-14-Take1'.
    basepath : str
        Path to dataset including set, e.g. '/dataset/orientation/training'.
    default_size : str
        Default image size to use. Should be either 'small' or 'large'.
    """
    def __init__(self, basename, basepath, default_size=DEFAULT_SIZE):
        self._basename = basename
        self._basepath = basepath

        assert default_size in SIZES
        self._default_size = default_size

        # determine person identifier
        res = re.findall('p([0-9]+)-', basename)
        assert len(res) == 1
        self._person_id = int(res[0])

        # determine set
        self._set_name = os.path.basename(basepath)
        assert self._set_name in SETS

        # determine json filepath
        self._json_filepath = self._build_filepath(_JSON_SUBFOLDER,
                                                   _JSON_SUFFIX)
        self._json = None

    def _build_filepath(self, subfolder, suffix):
        return os.path.join(self._basepath, subfolder, f"p{self._person_id}",
                            self._basename+suffix)

    @property
    def basename(self):
        return self._basename

    @property
    def basepath(self):
        return self._basepath

    @property
    def person_id(self):
        return self._person_id

    @property
    def set_name(self):
        return self._set_name

    def _none_to_default_size_decorator(f):
        @wraps(f)
        def wrapper(self, *args, **kwargs):
            if len(args) == 1:
                # given argument is the size argument
                args = (args[0] or self._default_size, )
                assert args[0] in SIZES
            elif len(kwargs) == 1:
                # given named argument is the size argument
                kwargs['size'] = kwargs['size'] or self._default_size
                assert kwargs['size'] in SIZES
            elif not args and not kwargs == 0:
                # size argument is not present at all
                kwargs['size'] = self._default_size
            return f(self, *args, **kwargs)
        return wrapper

    @_none_to_default_size_decorator
    def get_pcd_filepath(self, size=None):
        assert size == SIZE_SMALL
        return self._build_filepath('pcd', _PCD_SUFFIX)

    @_none_to_default_size_decorator
    def get_pcd(self, size=None):
        assert size == SIZE_SMALL
        raise NotImplementedError()

    @_none_to_default_size_decorator
    def get_depth_img_filepath(self, size=None):
        return self._build_filepath(size, _DEPTH_SUFFIX)

    @_none_to_default_size_decorator
    def get_depth_img(self, size=None):
        return img_utils.load(self.get_depth_img_filepath(size))

    @_none_to_default_size_decorator
    def get_depth_patch_filepath(self, size=None):
        return self._build_filepath(size + _PATCH_SUFFIX, _DEPTH_SUFFIX)

    @_none_to_default_size_decorator
    def get_depth_patch(self, size=None):
        filepath = self.get_depth_patch_filepath(size)
        if not os.path.exists(filepath):
            warnings.warn(f"Extracting patch from entire image since "
                          f"'{filepath}' does not exist. You should consider "
                          f"calling `extract_all_patches` first.")
            patch = Sample.extract_patch(self.get_depth_img(size),
                                         roi_y=self.get_roi_y(size),
                                         roi_x=self.get_roi_x(size))
        else:
            patch = img_utils.load(filepath)
        return patch

    @_none_to_default_size_decorator
    def get_mask_img_filepath(self, size=None):
        return self._build_filepath(size, _MASK_SUFFIX)

    @_none_to_default_size_decorator
    def get_mask_img(self, size=None):
        return img_utils.load(self.get_mask_img_filepath(size))

    @_none_to_default_size_decorator
    def get_mask_patch_filepath(self, size=None):
        return self._build_filepath(size + _PATCH_SUFFIX, _MASK_SUFFIX)

    @_none_to_default_size_decorator
    def get_mask_patch(self, size=None):
        filepath = self.get_mask_patch_filepath(size)
        if not os.path.exists(filepath):
            warnings.warn(f"Extracting patch from entire image since "
                          f"'{filepath}' does not exist. You should consider "
                          f"calling `extract_all_patches` first.")
            patch = Sample.extract_patch(self.get_mask_img(size),
                                         roi_y=self.get_roi_y(size),
                                         roi_x=self.get_roi_x(size))
        else:
            patch = img_utils.load(filepath)
        return patch

    @_none_to_default_size_decorator
    def get_rgb_img_filepath(self, size=None):
        return self._build_filepath(size, _RGB_SUFFIX)

    @_none_to_default_size_decorator
    def get_rgb_img(self, size=None):
        return img_utils.load(self.get_rgb_img_filepath(size))

    @_none_to_default_size_decorator
    def get_rgb_patch_filepath(self, size=None):
        return self._build_filepath(size + _PATCH_SUFFIX, _RGB_SUFFIX)

    @_none_to_default_size_decorator
    def get_rgb_patch(self, size=None):
        filepath = self.get_rgb_patch_filepath(size)
        if not os.path.exists(filepath):
            warnings.warn(f"Extracting patch from entire image since "
                          f"'{filepath}' does not exist. You should consider "
                          f"calling `extract_all_patches` first.")
            patch = Sample.extract_patch(self.get_rgb_img(size),
                                         roi_y=self.get_roi_y(size),
                                         roi_x=self.get_roi_x(size))
        else:
            patch = img_utils.load(filepath)
        return patch

    @property
    def json(self):
        if not self._json:
            self._json = read_json(self._json_filepath)
        return self._json

    @property
    def orientation(self):
        return self.json['Degree']

    @_none_to_default_size_decorator
    def get_roi_x(self, size=None):
        return self.json['ROI_x'][size]

    @_none_to_default_size_decorator
    def get_roi_y(self, size=None):
        return self.json['ROI_y'][size]

    @staticmethod
    def extract_patch(img, roi_y, roi_x):
        return img[slice(*roi_y), slice(*roi_x), ...]

    @classmethod
    def from_filepath(cls, filepath, default_size=DEFAULT_SIZE):
        """
        Instantiate the entire sample object from a single given filepath to
        one of the depth/mask/rgb/json file.

        Parameters
        ----------
        filepath : str
            Filepath where to derive the basename from.
        default_size : str
            Default image size to use. Should be either 'small' or 'large'.

        Returns
        -------
        sample : Sample
            The sample object.

        """
        # determine sample basename, e.g. p4-plain-Kinect1-14-Take1 from
        # /datasets/orientation/training/json/p4/p4-plain-Kinect1-14-Take1.json
        basename = _get_sample_basename(filepath)

        # determine basepath, e.g. /datasets/orientation/training from
        # /datasets/orientation/training/json/p4/p4-plain-Kinect1-14-Take1.json
        basepath = os.path.dirname(os.path.dirname(os.path.dirname(filepath)))

        return cls(basename=basename,
                   basepath=basepath,
                   default_size=default_size)


def load_set(dataset_basepath, set_name, default_size='small'):
    """
    Load a specific set of the dataset.

    Parameters
    ----------
    dataset_basepath : str
        Path to dataset root, e.g. '/dataset/orientation/'.
    set_name : str
        Set to load, should be one of 'training', 'validation' or 'test'.
    default_size : str
        Default image size to use. Should be either 'small' or 'large'.

    Returns
    -------
    dataset : RGBDOrientationDataset

    """
    return RGBDOrientationDataset(dataset_basepath, set_name, default_size)


def get_set_from_person_identifier(person_id):
    """
    Derive the associated set from a given person identifier.

    Parameters
    ----------
    person_id : {str, int}
        The person identifier to use for deriving the set. Can be a single
        `int` or a `str` with format 'pINT'.

    Returns
    -------
    set_ : str
        The associated set, one of 'training', 'validation' or 'test'.

    """
    if isinstance(person_id, str):
        person_id = int(person_id[1:])

    if person_id in TRAIN_PERSON_IDS:
        return TRAIN_SET
    elif person_id in VALID_PERSON_IDS:
        return VALID_SET
    elif person_id in TEST_PERSON_IDS:
        return TEST_SET

    raise ValueError(f"Unknown person identifier: '{person_id}''")


def _get_sample_basename(filepath):
    basename = os.path.basename(filepath)
    # remove possible suffixes
    for suffix in [_DEPTH_SUFFIX, _MASK_SUFFIX, _RGB_SUFFIX, _JSON_SUFFIX]:
        basename = basename.replace(suffix, '')
    return basename
