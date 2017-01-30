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
		void init(int input_dim, int output_dim);

		Matrix* weights;

		Layer* duplicateWithSameWeights();

		int getInputDim();
		int getOutputDim();

		bool hasSparseInput();
		bool hasSparseOutput();

		Vector* getOutput();
		Vector* getInputGrad();

		std::vector<int> getFullOutputIndices();

		int updateOutput(Vector* input, std::vector<int> &not_used);
		int updateInputGrad(Vector* output_grad);
		int accGradParameters(Vector* input, Vector* output_grad, float alpha);
};

class FlexLinear: public FlexLayer {

	private:

		int forward_code;

		bool weights_configured_input_sparse;

		int input_dim;
		int output_dim;

		bool output_must_be_sparse;
		bool input_must_be_sparse;

		bool mandatory_identical_input_output_sparsity;

		bool contains_layers;

		Vector* input;
		Vector* output;

		Vector* input_grad;
		Vector* output_grad;

		std::vector<int> input_indices;
		std::vector<int> output_indices;

	public:
		FlexLinear(int input_dim, int output_dim);
		FlexLinear(int input_dim, int output_dim, bool init_weights);
		// ~FlexLinear();
		void init(int input_dim, int output_dim, bool init_weights);

		int destroy(bool dont_destroy_weights);

		Matrix* weights;

		FlexLayer* duplicateWithSameWeights();

		int swapInputOutputSparsity();
		bool mandatoryIdenticalInputOutputSparsity();

		int getInputDim();
		int getOutputDim();

		bool inputMustBeSparse();
		bool outputMustBeSparse();

		bool containsLayers();

		Vector* getOutput();
		std::vector<int> &getOutputIndices();
		Vector* getInputGrad();
		int setOutputGrad(Vector* output_grad);

		// int updateOutput(Vector* input, std::vector<int> &not_used);
		int updateOutputDenseToDense(Vector* input);
		int updateOutputDenseToWeightedSparse(Vector* input, std::vector<int> &output_indices);
		int updateOutputWeightedSparseToDense(Vector* input, std::vector<int> &input_indices);
		int updateOutputWeightedSparseToWeightedSparse(Vector* input, std::vector<int> &input_indices, std::vector<int> &output_indices);
		int updateOutputBinarySparseToDense(std::vector<int> &input_indices);
		int updateOutputBinarySparseToWeightedSparse(std::vector<int> &input_indices, std::vector<int> &output_indices);

		int backward(Vector* output_grad);
		int updateInputGrad(Vector* output_grad);
		int accGradParameters(float alpha);

		int reset();
};

#endif