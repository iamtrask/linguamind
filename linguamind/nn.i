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
#include "nn/sequential.h"
%}

%include "cpointer.i"

/* Wrap a class interface around an "int *" */
%pointer_class(char, charp);

class SparseLinearInput {

	public:
		SparseLinearInput(int, int);

		Tensor* weights;
		Tensor* output;

		void init(int, int);

		void updateOutput(std::vector<int> input);
};

class SparseLinearOutput {

	public:
		SparseLinearOutput(int, int, int);

		Tensor* weights;
		Tensor* output;

		std::vector<int> output_indices;

		void init(int, int, int);

		void updateOutput(Tensor* input, std::vector<int> output_indices);
};

class Sequential {

	public:
		Sequential();
};

