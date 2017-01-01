/* File: linalg.i */
%module linalg

%{
#include <vector>
%}

%include "std_vector.i"

namespace std {
   %template(vectori) vector<int>;
   %template(vectorf) vector<float>;
   %template(vectord) vector<double>;
};

%{
#define SWIG_FILE_WITH_INIT
#include "linalg/tensor.h"
%}

%include "cpointer.i"

/* Wrap a class interface around an "int *" */
%pointer_class(char, charp);

class Tensor{

	public:
		Tensor();
		Tensor(std::vector<int> shape);
		std::vector<int> shape;
		int ndims;

		float* _data;
		long num_elements;
		unsigned long long seed;

		float dotRow(Tensor* a, int index);
		Tensor addRowi(Tensor* a, int index);

		Tensor uniform();
		Tensor zero();

		float get(int x);

		Tensor operator*=(float x) const;
		Tensor operator/=(float x) const;
		Tensor operator+=(float x) const;
		Tensor operator-=(float x) const;
};



