from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize("knn_cy.pyx", annotate=True, language_level=3),
    include_dirs=[np.get_include()]
)
