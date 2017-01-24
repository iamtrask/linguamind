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

		void init(int dim);

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

class FlexSigmoid: public FlexLayer {

	private:

		int input_dim;
		int output_dim;

		bool input_must_be_sparse;
		bool output_must_be_sparse;

		bool contains_layers;

		bool mandatory_identical_input_output_sparsity;

		int forward_code;

		Vector* output;
		Vector* input_grad;
		Vector* output_grad;

		std::vector<int> input_indices;
		std::vector<int> output_indices;

		float* expTable;

	public:
		
		FlexSigmoid(int dim);
		int destroy(bool dont_destroy_weights);
		
		FlexLayer* duplicateWithSameWeights();

		int getInputDim();
		int getOutputDim();

		bool inputMustBeSparse();
		bool outputMustBeSparse();

		bool containsLayers();

		bool mandatoryIdenticalInputOutputSparsity();

		Vector* getOutput();
		std::vector<int> &getOutputIndices();
		Vector* getInputGrad();
		int setOutputGrad(Vector* output_grad);

		int updateOutputDenseToDense(Vector* input);
		int updateOutputDenseToWeightedSparse(Vector* input, std::vector<int> &sparse_output);
		int updateOutputWeightedSparseToDense(Vector* input, std::vector<int> &sparse_input);
		int updateOutputWeightedSparseToWeightedSparse(Vector* input, std::vector<int> &sparse_input, std::vector<int> &sparse_output);
		int updateOutputBinarySparseToDense(std::vector<int> &sparse_input);
		int updateOutputBinarySparseToWeightedSparse(std::vector<int> &sparse_input, std::vector<int> &sparse_output);

		int backward(Vector* output_grad);
		int updateInputGrad(Vector* output_grad);
		int accGradParameters(float alpha);
};

#endif