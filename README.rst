|Travis|_ |Coveralls|_ |CircleCI|_

.. |Travis| image:: https://travis-ci.org/CEA-COSMIC/pysap-mri.svg?branch=master
.. _Travis: https://travis-ci.org/CEA-COSMIC/pysap-mri

.. |Coveralls| image:: https://coveralls.io/repos/CEA-COSMIC/pysap-mri/badge.svg?branch=master&kill_cache=1
.. _Coveralls: https://coveralls.io/github/CEA-COSMIC/pysap-mri

.. |CircleCI| image:: https://circleci.com/gh/CEA-COSMIC/pysap-mri.svg?style=svg
.. _CircleCI: https://circleci.com/gh/CEA-COSMIC/pysap-mri

pySAP-mri
===============

Python Sparse data Analysis Package external MRI plugin.

This work is made available by a community of people, amoung which the
CEA Neurospin UNATI and CEA CosmoStat laboratories, in particular A. Grigis,
J.-L. Starck, P. Ciuciu, and S. Farrens.

Installation instructions
===============

Install python-pySAP using `pip install python-pySAP`. Later install pysap-mri by calling setup.py
Note: If you want to use undecimated wavelet transform, please point the `$PATH` environment variable to
pysap external binary directory:

`export PATH=$PATH:/path-to-pysap/build/temp.linux-x86_64-<PYTHON_VERSION>/extern/bin/`

Important links
===============

- Official pySAP source code repo: https://github.com/cea-cosmic/pysap
- pySAP HTML documentation (last stable release): http://cea-cosmic.github.io/pysap
