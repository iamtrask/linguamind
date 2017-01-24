#ifndef LAYER
#define LAYER

#include "../linalg/seed.h"
#include "../linalg/vector.h"
#include "../linalg/matrix.h"

#include "layer.h"

class Layer {

	private:

		bool sparse_output;
		bool sparse_input;

		int input_dim;
		int output_dim;

		Vector* output;
		Vector* input_grad;

		std::vector<int> full_output_indices;

	public:
		
		Layer() {
    		
  		}
  		virtual ~Layer() {

		}

		virtual Layer* duplicateWithSameWeights() = 0;

		virtual int getInputDim() = 0;
		virtual int getOutputDim() = 0;

		virtual bool hasSparseInput() = 0;
		virtual bool hasSparseOutput() = 0;

		virtual Vector* getOutput() = 0;
		virtual Vector* getInputGrad() = 0;

		virtual std::vector<int> getFullOutputIndices() = 0;

		virtual int updateOutput(Vector*, std::vector<int> &sparse_output) = 0;
		virtual int updateInputGrad(Vector* output_grad) = 0;
		virtual int accGradParameters(Vector* input, Vector* output_grad, float alpha) = 0;
};


class FlexLayer {

	private:

		int input_dim;
		int output_dim;

		bool input_must_be_sparse;
		bool output_must_be_sparse;

		bool mandatory_identical_input_output_sparsity;

		bool contains_layers;

		Vector* output;
		Vector* input_grad;
		Vector* output_grad;

		std::vector<int> input_indices;
		std::vector<int> output_indices;

	public:
		
		FlexLayer() {
    		
  		}
  		virtual ~FlexLayer() {

		}

		virtual int destroy(bool dont_destroy_weights) = 0;

		virtual FlexLayer* duplicateWithSameWeights() = 0;

		virtual int getInputDim() = 0;
		virtual int getOutputDim() = 0;

		virtual bool inputMustBeSparse() = 0;
		virtual bool outputMustBeSparse() = 0;
		virtual bool mandatoryIdenticalInputOutputSparsity() = 0;

		virtual bool containsLayers() = 0;

		virtual Vector* getOutput() = 0;
		virtual std::vector<int> &getOutputIndices() = 0;
		virtual Vector* getInputGrad() = 0;
		virtual int setOutputGrad(Vector* output_grad) = 0;

		virtual int updateOutputDenseToDense(Vector* input) = 0;
		virtual int updateOutputDenseToWeightedSparse(Vector* input, std::vector<int> &sparse_output) = 0;
		virtual int updateOutputWeightedSparseToDense(Vector* input, std::vector<int> &sparse_input) = 0;
		virtual int updateOutputWeightedSparseToWeightedSparse(Vector* input, std::vector<int> &sparse_input, std::vector<int> &sparse_output) = 0;
		virtual int updateOutputBinarySparseToDense(std::vector<int> &sparse_input) = 0;
		virtual int updateOutputBinarySparseToWeightedSparse(std::vector<int> &sparse_input, std::vector<int> &sparse_output) = 0;

		virtual int backward(Vector* output_grad) = 0;
		virtual int updateInputGrad(Vector* output_grad) = 0;
		virtual int accGradParameters(float alpha) = 0;
};
#endif