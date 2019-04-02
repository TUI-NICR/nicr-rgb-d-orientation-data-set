# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>

"""
from .dataset import TRAIN_SET
from .dataset import VALID_SET
from .dataset import TEST_SET
from .dataset import SETS

from .dataset import TRAIN_PERSON_IDS
from .dataset import VALID_PERSON_IDS
from .dataset import TEST_PERSON_IDS

from .dataset import SIZE_SMALL
from .dataset import SIZE_LARGE
from .dataset import SIZES
from .dataset import DEFAULT_SIZE

from .dataset import RGBDOrientationDataset
from .dataset import Sample

from .dataset import load_set
from .dataset import get_set_from_person_identifier

from .version import __version__
