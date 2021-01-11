Overview
=============

Code for tomographic reconstruction built around the ASTRA toolbox (https://www.astra-toolbox.com/). This code implements model-based computed tomography (CT) reconstruction algorithms based on a Markov random field (MRF) prior and requires systems with atleast one GPU. The algorithms involve formulating and finding a minimum of the cost function of the form:

```math
c(x) = l(y;Ax)+r(x)
```

where $`l(y;Ax)`$ is a data-fidelity term, $`A`$ is the tomographic projector and $`r(x)`$ is a regularizer based on the q-generalized Markov random field [1]. Currently, the package supports parallel beam tomography, laminography [2], arbitary view CT with a point-spread function as in cryo-EM [3] and conventional cone-beam CT for the choice of the $`A`$ matrix (see examples folder). 

Requirements
=============

astra toolbox : Core GPU based projection and back-projection

numpy, scipy, matplotlib, time, gc, concurrent, psutil, ctypes

pyqtgraph (optional): For displaying 3D volumes

gcc

Installation
=============

1) Install the above packages

2) Compile the code in source/prior_model folder by cd-ing and running make

3) Add the pyMBIR directory to your PYTHONPATH 

Getting started
=============

We highly recommend starting with the examples in the examples/sim directory to develop an understanding of the pyMBIR package and how to set the different parameters.   

References
=============

[1] Thibault, Jean‐Baptiste, et al. "A three‐dimensional statistical approach to improved image quality for multislice helical CT." Medical physics 34.11 (2007): 4526-4544.

[2] Venkatakrishnan, Singanallur V., et al. "Model-based iterative reconstruction for neutron laminography." 2017 51st Asilomar Conference on Signals, Systems, and Computers. IEEE, 2017.

[3] Singanallur Venkatakrishnan, Puneet Juneja, Hugh O’Neill, “Model-based Reconstruction for Single Particle Cryo-Electron Microscopy”, Proc. of IEEE Asilomar Conference of Signals, Systems and Computer 2020


License
============

pyMBIR is distributed as open-source software under a GPL License (see the LICENSE file for details)
