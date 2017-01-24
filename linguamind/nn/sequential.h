#ifndef SEQUENTIAL
#define SEQUENTIAL

#include "../linalg/vector.h"
#include "../linalg/matrix.h"
#include "sparse_linear.h"
#include "layer.h"
#include <vector>
#include <iostream>
#include <memory>

class Sequential  {

	public:
		Sequential(std::vector<Layer*> layers);

		std::vector<Layer*> layers;
		Vector* output;

		Sequential* duplicateWithSameWeights();

		Layer* get(int i);
		Vector* forward(std::vector<int> &input_indices,std::vector<int> &output_indices);
		void backward(Vector* grad, std::vector<int> &output_indices);
};

class FlexSequential: public FlexLayer  {

	private:
		
		bool output_must_be_sparse;
		bool input_must_be_sparse;

		bool contains_layers;

		bool mandatory_identical_input_output_sparsity;
		int layer_index_to_begin_using_sequence_output_indices;

		int forward_code;

		int input_dim;
		int output_dim;

		Vector* input;
		Vector* output;

		Vector* input_grad;
		Vector* output_grad;

		std::vector<int> input_indices;
		std::vector<int> output_indices;

		std::vector<int> full_input_indices;
		std::vector<int> full_output_indices;

		std::vector<int> not_used;

		
		int num_layers;

	public:

		std::vector<FlexLayer*> layers;

		FlexSequential(std::vector<FlexLayer*> layers);
		void init(int input_dim, int output_dim);

		FlexLayer* duplicateWithSameWeights();
		FlexSequential* duplicateSequentialWithSameWeights();

		int getLayerIndexToBeginUsingSequenceOutputIndices();

		FlexLayer* get(int i);

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

		std::vector<int> getFullOutputIndices();

		Vector* forward(std::vector<int> &input_indices,std::vector<int> &output_indices);

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

};


#endif