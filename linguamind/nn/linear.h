#ifndef LINEAR
#define LINEAR

#include <vector>
#include "../linalg/vector.h"
#include "../linalg/matrix.h"

#include "layer.h"

class Linear: public Layer {

	private:
		
		bool sparse_output;
		bool sparse_input;

		int input_dim;
		int output_dim;

		Vector* output;
		Vector* input_grad;

		std::vector<int> full_output_indices;

	public:
		Linear(int input_dim, int output_dim);

		Matrix* weights;

		int getInputDim();
		int getOutputDim();

		bool hasSparseInput();
		bool hasSparseOutput();

		Vector* getOutput();
		Vector* getInputGrad();

		std::vector<int> getFullOutputIndices();

		int updateOutput(Vector* input, std::vector<int> not_used);
		int updateInputGrad(Vector* output_grad);
		int accGradParameters(Vector* input, Vector* output_grad, float alpha);
};

#endif