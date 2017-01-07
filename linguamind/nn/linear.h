#ifndef LINEAR
#define LINEAR

#include <vector>
#include "../linalg/tensor.h"
#include "layer.h"

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

#endif