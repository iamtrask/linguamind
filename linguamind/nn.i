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
#include "nn/layer.h"
#include "nn/sparse_linear.h"
#include "nn/sequential.h"
#include "nn/criterion.h"
%}

%include "cpointer.i"

/* Wrap a class interface around an "int *" */
%pointer_class(char, charp);

class Layer {

	public:
		Layer();

		Tensor* weights;
		Tensor* output;
};

class SparseLinearInput: public Layer {

	public:
		SparseLinearInput(int, int);

		Tensor* weights;
		Tensor* output;

		void init(int, int);

		void updateOutput(std::vector<int> input);
};

class SparseLinearOutput: public Layer {

	public:
		SparseLinearOutput(int, int, int);

		Tensor* weights;
		Tensor* output;

		std::vector<int> output_indices;

		void init(int, int, int);

		void updateOutput(Tensor* input, std::vector<int> output_indices);
};

class Sequential  {

	public:
		Sequential();

		std::vector<Layer*> layers;
		Tensor* output;

		void add(Layer* layer);
		Tensor* forward(std::vector<int> input_indices,std::vector<int> output_indices);
};

class MSECriterion {

	public:
		int batch_size;
		int dim;

		Tensor* output;
		Tensor* grad_input;
	
		MSECriterion(int batch_size, int dim);
		
		void forwards(Tensor* input);
		Tensor* backwards(Tensor* target);
};

