# setup.py

from distutils.core import setup, Extension


example_module = Extension('_example', sources=['example_wrap.cxx', 'example.cxx'])

setup(name='example', ext_modules=[example_module], py_modules=["example"])