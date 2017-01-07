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
#include "linalg/seed.h"
#include "linalg/vector.h"
#include "linalg/matrix.h"
#include "nn/layer.h"
#include "nn/sparse_linear.h"
#include "nn/linear.h"
%}

%include "cpointer.i"

/* Wrap a class interface around an "int *" */
%pointer_class(char, charp);

class Layer {

	public:
		bool sparse_output;
		bool sparse_input;

		int input_dim;
		int output_dim;

		Vector* output;
		Vector* input_grad;

		std::vector<int> full_output_indices;
};

class SparseLinearInput: public Layer {

	public:
		SparseLinearInput(int input_dim, int output_dim);

		bool sparse_output;
		bool sparse_input;

		int input_dim;
		int output_dim;

		Vector* output;
		Vector* input_grad;

		Matrix* weights;
		std::vector<int> input_indices;
		std::vector<int> full_output_indices;

		void updateOutput(Vector* input, std::vector<int> input_indices);
		void updateInputGrad(Vector* output_grad);
		void accGradParameters(Vector* input, Vector* output_grad, float alpha);
};

class SparseLinearOutput: public Layer {

	public:
		SparseLinearOutput(int input_dim, int output_dim);

		bool sparse_output;
		bool sparse_input;

		int input_dim;
		int output_dim;

		Vector* output;
		Vector* input_grad;

		Matrix* weights;
		std::vector<int> output_indices;
		std::vector<int> full_output_indices;

		void updateOutput(Vector* input, std::vector<int> output_indices);
		void updateInputGrad(Vector* output_grad);
		void accGradParameters(Vector* input, Vector* output_grad, float alpha);
};

class Linear: public Layer {

	public:
		Linear(int input_dim, int output_dim);

		bool sparse_output;
		bool sparse_input;

		int input_dim;
		int output_dim;

		Vector* output;
		Vector* input_grad;

		Matrix* weights;

		std::vector<int> full_output_indices;

		void updateOutput(Vector* input, std::vector<int> not_used);
		void updateInputGrad(Vector* output_grad);
		void accGradParameters(Vector* input, Vector* output_grad, float alpha);
};
