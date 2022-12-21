from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    name="test",
    ext_modules=cythonize("cy_spurious_mcts.pyx"),
    # ext_modules=cythonize("cy_mini_game.pyx"),
    include_dirs=[numpy.get_include()],
)
