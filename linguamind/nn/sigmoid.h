#ifndef SIGMOID
#define SIGMOID

#include <vector>
#include "../linalg/vector.h"
#include "../linalg/matrix.h"

#include "layer.h"

class Sigmoid: public Layer {
	
	private:
		
		bool sparse_output;
		bool sparse_input;

		int input_dim;
		int output_dim;

		Vector* output;
		Vector* input_grad;

		Matrix* weights;

		std::vector<int> output_indices;
		std::vector<int> full_output_indices;

	public:
		Sigmoid(int dim);

		int getInputDim();
		int getOutputDim();

		bool hasSparseInput();
		bool hasSparseOutput();

		Vector* getOutput();
		Vector* getInputGrad();

		std::vector<int> getFullOutputIndices();

		int updateOutput(Vector* input, std::vector<int> &output_indices);
		int updateInputGrad(Vector* output_grad);
		int accGradParameters(Vector* input, Vector* output_grad, float alpha);
};

#endif