export CC=/usr/bin/clang
export CXX=/usr/bin/clang++

sudo rm -rf build
touch linguamind/__init__.py

# swig -c++ -python linguamind/nlp.i
swig -c++ -python linguamind/linalg.i
swig -c++ -python linguamind/nn.i

# python setup.py build_ext --inplace
sudo python setup.py clean install

rm -rf linguamind/*.cxx
rm -rf linguamind/*.py