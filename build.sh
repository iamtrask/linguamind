export CC=/usr/bin/clang
export CXX=/usr/bin/clang++

sudo rm -rf build
rm linguamind.py
rm linguamind.pyc
rm linguamind_wrap.c
rm linguamind_wrap.cxx
rm _linguamind.so

# swig -python linguamind.i
swig -c++ -python linguamind.i

# python setup.py build_ext --inplace
sudo python setup.py install