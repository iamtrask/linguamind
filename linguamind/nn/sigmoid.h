#ifndef SIGMOID
#define SIGMOID

#include <vector>
#include <math.h>

#include "../linalg/vector.h"
#include "../linalg/matrix.h"

#include "layer.h"

#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6

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

		float* expTable;

	public:
		Sigmoid(int dim);

		Layer* duplicateWithSameWeights();

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