.. myforestplot documentation master file, created by
   sphinx-quickstart on Mon Sep  5 15:18:25 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. |br| raw:: html

    <br/>

.. |nbsp|   unicode:: U+00A0 .. NO-BREAK SPACE

MyForestPlot documentation
========================================

**Date**: |today| |nbsp|   **Version**: |release| |br|
**Useful links**: |nbsp| `Source Repository <https://github.com/toshiakiasakura/myforestplot>`_
|nbsp| | |nbsp| `Issues & Ideas <https://github.com/toshiakiasakura/myforestplot/issues>`_

What is MyForestPlot? 
-----------------------
MyForestPlot is a Python package helping create a forest plot. This myforestplot is 
mainly designed to create a forest plot for logistic/log binomial/robust poisson results, 
meaning that tries to present ORs or RRs for categorical variables.
Codes and usage is explained in :doc:`usage page<notebooks/1_quickstart>`.

Installation
--------------------------------

.. code-block:: bash

   pip install myforestplot


.. toctree::
   :maxdepth: 2
   :caption: How to use:

   notebooks/1_quickstart
   notebooks/3_architecture
   notebooks/4_practical_examples

.. toctree::
   :maxdepth: 3
   :caption: Gallery:

   notebooks/2_gallery


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
