#!/usr/bin/env python

"""
setup.py file for SWIG linguamind
"""

from distutils.core import setup, Extension


nlp_module = Extension('_nlp',
                           sources=['linguamind/nlp_wrap.cxx', 'linguamind/nlp/text.cpp', 'linguamind/nlp/vocab.cpp'],
                           include_dirs = ['/usr/local/opt/openblas/include/'],
                           extra_compile_args = ["-std=c++11", "-stdlib=libc++", "-mmacosx-version-min=10.7", "-pthread", "-O3", "-march=native", "-Wall", "-funroll-loops", "-Wno-unused-result","-I/usr/local/opt/openblas/include/"],
                           )

linalg_module = Extension('_linalg',
                           sources=['linguamind/linalg_wrap.cxx', 'linguamind/linalg/seed.cpp','linguamind/linalg/vector.cpp','linguamind/linalg/matrix.cpp', 'linguamind/linalg/tensor.cpp'],
                           include_dirs = ['/usr/local/opt/openblas/include/'],
                           libraries = ['cblas'],
                           extra_compile_args = ["-std=c++11", "-stdlib=libc++", "-mmacosx-version-min=10.7", "-pthread", "-O3", "-march=native", "-Wall", "-funroll-loops", "-Wno-unused-result","-I/usr/local/opt/openblas/include/"],
                           )

nn_module = Extension('_nn',
                           sources=['linguamind/nn_wrap.cxx','linguamind/nn/layer.cpp', 'linguamind/nn/linear.cpp','linguamind/nn/lstm.cpp' ,'linguamind/nn/sparse_linear.cpp', 'linguamind/nn/relu.cpp', 'linguamind/nn/sigmoid.cpp' ,'linguamind/nn/tanh.cpp', 'linguamind/nn/criterion.cpp', 'linguamind/nn/sequential.cpp', 'linguamind/nn/training_generators.cpp', 'linguamind/nn/stochastic_gradient.cpp','linguamind/nn/hierarchical_layers.cpp'],
                           include_dirs = ['/usr/local/opt/openblas/include/'],
                           extra_compile_args = ["-std=c++11", "-stdlib=libc++", "-mmacosx-version-min=10.7", "-pthread", "-O3", "-march=native", "-Wall", "-funroll-loops", "-Wno-unused-result","-I/usr/local/opt/openblas/include/"],
                           )

setup (name = 'linguamind',
       version = '0.1',
       author      = "SWIG Docs",
       description = """Simple swig linguamind from docs""",
       ext_modules = [nlp_module, linalg_module, nn_module],
       py_modules = ["linguamind.nlp","linguamind.linalg", "linguamind.nn"],
       )