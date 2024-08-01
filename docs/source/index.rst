.. Vipdopt documentation master file, created by
   sphinx-quickstart on Thu Jun 13 13:39:45 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Vipdopt's documentation!
===================================

* GitHub: https://github.com/Faraon-Lab/vidpopt
* PyPi: https://pypi.org/project/vipdopt/
* Documentation: https://vidpopt.readthedocs.io/en/latest/

``vipdopt`` (/vɪpdɑpt/) is a Python library for streamlining the process of **v**\olumetric
**i**\nverse **p**\hotonic **d**\esign and **opt**\imization. It
makes use of Lumerical's Python API ``lumapi`` to make the design and optimization of new
optical devices more straightforward.

This code performs an inverse-design optimization based on the adjoint optimization
technique [#]_ that designs a freeform multilayer optical Bayer filter and focusing lens [#]_.
The Bayer filter is multiwavelength and focuses and sorts 
different spectral bands into different prescribed locations on the focal plane. 
Otherwise known as a color router, this code is set by default to create a 10-layer 
filter for the visible spectral range, with lateral and vertical dimensions of 2.04 
microns. Different options are available in the config to, for example, enable 
polarization sorting or adjust the spectral bands as necessary.


.. [#] https://doi.org/10.1364/OPTICA.384228
.. [#] https://doi.org/10.1364/OE.21.021693

Check out the :doc:`usage` section for further information.





.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. role:: underline
    :class: underline

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


Contents
--------

.. toctree::
    :maxdepth: 3

    installation
    usage
    notebooks
    config
    api

