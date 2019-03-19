Repository of "EUSIPCO" paper:
##############################


Authors
-------

Hamza Cherkaoui
Benoit Sarthou
Philippe Ciuciu


Synopsis
--------

This repository is dedicated to reproduce the results in the
"EUSIPCO" paper.


Dependencies
------------

* pysap


Instructions
------------

Create a configuration file:

.. code-block:: bash

    gedit config.ini

Launch the multiple reconstructions - gridsearch (long long time...):

.. code-block:: bash

    pysap_gridsearch -h

Produce the plots:

.. code-block:: bash

    pysap_gridsearch_report -h


Create Configuration
--------------------

Please find below an example of configuration file:

.. code-block:: bash

    # The Global section is dedicated to global option that will be used for each
    # reconstruction.

    [Global]
    # n_jobs: -1 correspond to all the cpu, -2 correspond to all the cpu minus one,
    # any other postif integer correspond to the desired number of cpu.
    n_jobs: 8

    # timeout: the time out option for each reconstruction in second.
    # usual values 600=10min, 1200=20min, 1800=30min, 2100=35min, 9999=2h45
    timeout: 9999

    # max_nb_of_iter: the maximum number of iteration for each reconstruction.
    max_nb_of_iter: 200

    # verbose_reconstruction: the verbosity for each reconstruction.
    verbose_reconstruction: 0

    # verbose_gridsearch: the verbosity for each gridsearch function.
    verbose_gridsearch: 11


    #------------------------------------------------------------------------------


    # How to declare a new run, a simple example:
    #
    # # Be carefull: the name of the run section should have 'Run' in it.
    # [Run1]

    # # Be carefull: the name of the kspace sampling trajectory should be one of the
    # # available in the data module: cartesianR4, sparkling or radial.
    # mask_type: cartesianR4

    # # the available acceleration factor dependent of the choice of the mask type,
    # # if there is no need for that option write None.
    # acc_factor: None

    # # the sigma correspond to the standard deviation of the centered gaussian
    # # noise added to the kspace. If only one value is desired, put it
    # # in a list: [0.0]
    # sigma: [0.0, 0.1, 0.2, 0.4, 0.6, 0.8]
    #

    [Run1]
    mask_type: cartesianR4
    acc_factor: None
    sigma: [0.1, 0.2]

    # # [Run1]
    # # mask_type: cartesianR4
    # # acc_factor: None
    # # sigma: [0.0, 0.1, 0.2, 0.4, 0.6, 0.8]

    # # [Run2]
    # # mask_type: radial-sparkling
    # # acc_factor: 8
    # # sigma: [0.0, 9.0e-6, 2.0e-5, 5.0e-5, 8.0e-5]

    # # [Run3]
    # # mask_type: radial
    # # acc_factor: 8
    # # sigma: [0.0, 4.0e-5, 6.0e-5, 9.0e-5, 3.0e-4]

    # # [Run4]
    # # mask_type: radial-sparkling
    # # acc_factor: 15
    # # sigma: [0.0, 9.0e-6, 2.0e-5, 5.0e-5, 8.0e-5]

    # # [Run5]
    # # mask_type: radial
    # # acc_factor: 15
    # # sigma: [0.0, 4.0e-5, 6.0e-5, 9.0e-5, 3.0e-4]
