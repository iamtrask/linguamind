#!/usr/bin/env python

"""
setup.py file for SWIG linguamind
"""

from distutils.core import setup, Extension


linguamind_module = Extension('_linguamind',
                           sources=['linguamind_wrap.c', 'linguamind.cpp'],
                           )

setup (name = 'linguamind',
       version = '0.1',
       author      = "SWIG Docs",
       description = """Simple swig linguamind from docs""",
       ext_modules = [linguamind_module],
       py_modules = ["linguamind"],
       )