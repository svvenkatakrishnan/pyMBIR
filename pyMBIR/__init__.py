# Copyright (C) 2019, S.V.Venkatakrishnan <venkatakrisv@ornl.gov>
# All rights reserved. GPL v3 License.
# This file is part of the pyMBIR package. Details of the copyright
# and user license can be found in the 'LICENSE' file distributed
# with the package.

import os, ctypes
here = os.path.dirname(__file__)
prior_dir = os.path.join(here, 'prior_model')
lib = ctypes.cdll.LoadLibrary(os.path.join(prior_dir, 'mrf3d_grad_linear.so'))

