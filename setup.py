#!/usr/bin/env python

"""
setup.py file for SWIG linguamind
"""

from distutils.core import setup, Extension


# nlp_module = Extension('_nlp',
#                            sources=['linguamind/nlp_wrap.cxx', 'linguamind/nlp/text.cpp', 'linguamind/nlp/vocab.cpp'],
#                            )

linalg_module = Extension('_linalg',
                           sources=['linguamind/linalg_wrap.cxx', 'linguamind/linalg/seed.cpp','linguamind/linalg/vector.cpp','linguamind/linalg/matrix.cpp', 'linguamind/linalg/tensor.cpp'],
                           )

nn_module = Extension('_nn',
                           sources=['linguamind/nn_wrap.cxx', 'linguamind/nn/layer.cpp', 'linguamind/nn/linear.cpp' ,'linguamind/nn/sparse_linear.cpp'],
                           )

setup (name = 'linguamind',
       version = '0.1',
       author      = "SWIG Docs",
       description = """Simple swig linguamind from docs""",
       ext_modules = [linalg_module, nn_module],
       py_modules = ["linguamind.nlp", "linguamind.linalg", "linguamind.nn"],
       )