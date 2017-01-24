#ifndef SPARSE_LINEAR
#define SPARSE_LINEAR

#include <vector>
#include "../linalg/vector.h"
#include "../linalg/matrix.h"
#include "layer.h"

class SparseLinearInput: public Layer {

	private:

		bool sparse_output;
		bool sparse_input;

		int input_dim;
		int output_dim;

		Vector* output;
		Vector* input_grad;

		std::vector<int> input_indices;
		std::vector<int> full_output_indices;

	public:

		SparseLinearInput(int input_dim, int output_dim);

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

		int updateOutput(Vector* input, std::vector<int> &input_indices);
		int updateInputGrad(Vector* output_grad);
		int accGradParameters(Vector* input, Vector* output_grad, float alpha);
};

class SparseLinearOutput: public Layer {

	private:

		bool sparse_output;
		bool sparse_input;

		int input_dim;
		int output_dim;

		Vector* output;
		Vector* input_grad;

		std::vector<int> output_indices;
		std::vector<int> full_output_indices;

	public:

		SparseLinearOutput(int input_dim, int output_dim);
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

		int updateOutput(Vector* input, std::vector<int> &output_indices);
		int updateInputGrad(Vector* output_grad);
		int accGradParameters(Vector* input, Vector* output_grad, float alpha);
};

class WeightedSparseLinearInput: public Layer {

	private:

		bool sparse_output;
		bool sparse_input;

		int input_dim;
		int output_dim;

		Vector* output;
		Vector* input_grad;

		std::vector<int> input_indices;
		std::vector<int> full_output_indices;

	public:

		WeightedSparseLinearInput(int input_dim, int output_dim);

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
		
		int predict(Vector* input, std::vector<int> input_indices);
		int updateOutput(Vector* input, std::vector<int> &input_indices);
		int updateInputGrad(Vector* output_grad);
		int accGradParameters(Vector* input, Vector* output_grad, float alpha);
};

// class NegativeSamplingOutput: public Layer {

// 	private:
// 		bool sparse_output;
// 		bool sparse_input;

// 		int input_dim;
// 		int output_dim;

// 		Vector* output;
// 		Vector* input_grad;

// 		std::vector<int> output_indices;
// 		std::vector<int> full_output_indices;

// 		int negative_sample_size;
// 		int vocab_size;
// 		unsigned long long neg_pos;

// 	public:

// 		NegativeSamplingOutput(int input_dim, int negative_sample_size, int vocab_size);

// 		Matrix* weights;

// 		int getInputDim();
// 		int getOutputDim();

// 		bool hasSparseInput();
// 		bool hasSparseOutput();

// 		Vector* getOutput();
// 		Vector* getInputGrad();

// 		std::vector<int> getFullOutputIndices();
		
// 		int getRandom(int target);
// 		int updateOutput(Vector* input, std::vector<int> &output_indices);
// 		int updateInputGrad(Vector* output_grad);
// 		int accGradParameters(Vector* input, Vector* output_grad, float alpha);
// };

#endif