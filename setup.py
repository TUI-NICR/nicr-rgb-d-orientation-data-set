# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>

"""
import os
from setuptools import setup, find_packages


def run_setup():
    # get version
    version_namespace = {}
    with open(os.path.join('nicr_rgb_d_orientation_data_set',
                           'version.py')) as version_file:
        exec(version_file.read(), version_namespace)
    version = version_namespace['_get_version'](with_suffix=False)

    # setup
    setup(name='nicr_rgb_d_orientation_data_set',
          version='{}.{}.{}'.format(*version),
          description='NICR RGB-D Orientation Data Set',
          url='https://www.tu-ilmenau.de/neurob/data-sets-code/rgb-d-orientation/',
          author='Daniel Seichter',
          author_email='daniel.seichter@tu-ilmenau.de',
          license=('Copyright 2019, Neuroinformatics and Cognitive Robotics '
                   'Lab TU Ilmenau, Ilmenau, Germany'),
          python_requires='>3.6.',
          install_requires=[
              'numpy>=1.11.1',
          ],
          packages=find_packages(),
          extras_require={
              'test': [
                  'pytest>=3.0.2'
              ],
              'with_opencv': [
                  'opencv-python==3.4.2.*'
              ]
          })


if __name__ == '__main__':
    run_setup()
