# This file is part of rddl2tf.

# rddl2tf is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# rddl2tf is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with rddl2tf. If not, see <http://www.gnu.org/licenses/>.


import os
from setuptools import setup, find_packages

import rddl2tf


def read(filename):
    filepath = os.path.join(os.path.dirname(__file__), filename)
    file = open(filepath, 'r')
    return file.read()


setup(
    name='rddl2tf',
    version=rddl2tf.__version__,
    author='Thiago P. Bueno',
    author_email='thiago.pbueno@gmail.com',
    description='RDDL2TensorFlow compiler.',
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    license='GNU General Public License v3.0',
    keywords=['rddl', 'tensorflow', 'compiler'],
    url='https://github.com/thiagopbueno/rddl2tf',
    packages=find_packages(),
    scripts=['scripts/rddl2tf'],
    python_requires='>=3.6',
    install_requires=[
        'tensorflow>=2.0',
        'typing',
        'pyrddl',
        'rddlgym'
    ],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
)
