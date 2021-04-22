|Travis|_ |Coveralls|_ |Doc|_ |CircleCI|_

.. |Travis| image:: https://travis-ci.org/CEA-COSMIC/pysap-mri.svg?branch=master
.. _Travis: https://travis-ci.org/CEA-COSMIC/pysap-mri

.. |Coveralls| image:: https://coveralls.io/repos/CEA-COSMIC/pysap-mri/badge.svg?branch=master&kill_cache=1
.. _Coveralls: https://coveralls.io/github/CEA-COSMIC/pysap-mri

.. |Doc| image:: https://readthedocs.org/projects/pysap-mri/badge/?version=latest
.. _Doc: https://pysap-mri.readthedocs.io/en/latest/?badge=latest

.. |CircleCI| image:: https://circleci.com/gh/CEA-COSMIC/pysap-mri.svg?style=svg
.. _CircleCI: https://circleci.com/gh/CEA-COSMIC/pysap-mri

pySAP-mri
=========

Python Sparse data Analysis Package external MRI plugin.

This work is made available by a community of people, amoung which the
CEA Neurospin UNATI and CEA CosmoStat laboratories. This plugin was developed by
A. Grigis, J.-L. Starck, P. Ciuciu, and S. Farrens. Currently, it is being handled by
Chaithya G R, Z. Ramzi and S. Farrens.

Installation instructions
=========================

Install python-pySAP using:

``pip install python-pysap``

This will install pysap-mri plugin too. However if you want to update to a more
later version, please use:

``pip install --upgrade pysap-mri``

Special Installations
=====================

`pyNFFT <https://github.com/pyNFFT/pyNFFT/>`_
--------------------------------------------

For using Non-Cartesian FFT, please install pyNFFT by:

Linux:
``````

``sudo apt install libnfft3-dev``

``pip install pynfft2``

Mac:
````

``conda install -c conda-forge pynfft``

`gpuNUFFT <https://www.opensourceimaging.org/project/gpunufft/>`_
---------------------------------------------------------------

For faster NUFFT operation, pysap-mri uses gpuNUFFT, to run the NUFFT on GPU. To install gpuNUFFT, please use:

``pip install gpuNUFFT``

We are still in the process of merging this and getting a pip release for gpuNUFFT. Note, that you can still use CPU
NFFT without installing the package.

Important links
===============

- Official pySAP source code repo: https://github.com/cea-cosmic/pysap
- pySAP HTML documentation (last stable release): http://cea-cosmic.github.io/pysap
