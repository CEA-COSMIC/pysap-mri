# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# Module current version
version_major = 0
version_minor = 3
version_micro = 0

# Expected by setup.py: string of form "X.Y.Z"
__version__ = "{0}.{1}.{2}".format(version_major, version_minor, version_micro)

# Expected by setup.py: the status of the project
CLASSIFIERS = ["Development Status :: 1 - Planning",
               "Environment :: Console",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

# Project descriptions
description = """
Python Sparse data Analysis Package external MRI plugin.
"""
SUMMARY = """
.. container:: summary-carousel

    Python Sparse data Analysis Package external MRI plugin.
"""
long_description = """
Python Sparse data Analysis Package external MRI plugin.
"""

# Main setup parameters
NAME = "pysap-mri"
ORGANISATION = "CEA"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
EXTRANAME = "COSMIC webPage"
EXTRAURL = "http://cosmic.cosmostat.org/"
URL = "https://github.com/CEA-COSMIC/pysap-mri"
DOWNLOAD_URL = "https://github.com/CEA-COSMIC/pysap-mri"
LICENSE = "CeCILL-B"
CLASSIFIERS = CLASSIFIERS
AUTHOR = """
Antoine Grigis
Samuel Farrens
Jean-Luc Starck
Philippe Ciuciu
"""
AUTHOR_EMAIL = """
<antoine.grigis@cea.fr>
<samuel.farrens@cea.fr>
<jl.stark@cea.fr>
<philippe.ciuciu@cea.fr>
"""
PLATFORMS = "Linux,OSX"
ISRELEASE = True
VERSION = __version__
