export CC=/usr/bin/clang
export CXX=/usr/bin/clang++

rm -rf build
rm example.py
rm example.pyc
rm example_wrap.c
rm example_wrap.cxx
rm _example.so

# swig -python example.i
# swig -c++ -python example.i

# # python setup.py build_ext --inplace
# sudo python setup.py install