#! /usr/bin/env python
##########################################################################
# pySAP - Copyright (C) CEA, 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
from __future__ import print_function
import os
from setuptools import setup, find_packages

# Global parameters
CLASSIFIERS = [
    "Development Status :: 1 - Planning",
    "Environment :: Console",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Topic :: Scientific/Engineering"]
AUTHOR = """
Chaithya G R <chaithya.giliyarradhakrishna@cea.fr>
Pierre-Antoine Comby <pierre-antoine.comby@cea.fr>
Jean-Luc Starck <jl.stark@cea.fr>
Philippe Ciuciu <philippe.ciuciu@cea.fr>
"""
# Write setup
setup_requires = ["numpy>=1.16.4", "cython>=0.27.3", "pytest-runner"]

setup(
    name="pysap-mri",
    description="Python Sparse data Analysis Package external MRI plugin.",
    long_description="Python Sparse data Analysis Package external MRI plugin.",
    license="CeCILL-B",
    classifiers="CLASSIFIERS",
    author=AUTHOR,
    author_email="XXX",
    version="0.4.1",
    url="https://github.com/CEA-COSMIC/pysap-mri",
    packages=find_packages(),
    setup_requires=setup_requires,
    install_requires=[
        "scikit-learn>=0.19.1",
        "progressbar2>=3.34.3",
        "joblib>=1.0.0",
        "scipy>=1.3.0",
        "scikit-image>=0.17.0",
    ],
    tests_require=['pytest>=5.0.1', 'pytest-cov>=2.7.1', 'pytest-pep8', 'pytest-runner'],
    platforms="OS Independent"
)
