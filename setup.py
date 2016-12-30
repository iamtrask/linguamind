#!/usr/bin/env python

"""
setup.py file for SWIG linguamind
"""

from distutils.core import setup, Extension


nlp_module = Extension('_nlp',
                           sources=['linguamind/nlp_wrap.cxx', 'linguamind/nlp/text.cpp'],
                           )

linalg_module = Extension('_linalg',
                           sources=['linguamind/linalg_wrap.cxx', 'linguamind/linalg/matrix.cpp'],
                           )

setup (name = 'linguamind',
       version = '0.1',
       author      = "SWIG Docs",
       description = """Simple swig linguamind from docs""",
       ext_modules = [nlp_module, linalg_module],
       py_modules = ["linguamind.nlp", "linguamind.linalg"],
       )