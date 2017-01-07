#ifndef SPARSE_LINEAR
#define SPARSE_LINEAR

#include <vector>
#include "../linalg/tensor.h"
#include "layer.h"

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

#endif