/* File: nn.i */
%module nn

%{
#include <vector>
%}

%include "std_vector.i"

namespace std {
   %template(vectori) vector<int>;
   %template(vectord) vector<double>;
};

%{
#define SWIG_FILE_WITH_INIT
#include "linalg/tensor.h"
#include "nn/sparse_linear.h"
%}

%include "cpointer.i"

/* Wrap a class interface around an "int *" */
%pointer_class(char, charp);

class SparseLinear{

	public:
		SparseLinear(int, int);
		SparseLinear(int, int, bool);

		Tensor* weights;
		bool is_output_sparse;

		void init(int, int, bool);
};
