import distutils.core
import cython
import Cython.Build
distutils.core.setup(
    ext_modules = Cython.Build.cythonize("multprocessing_beamformer.pyx"))