#ifndef SPARSE_LINEAR
#define SPARSE_LINEAR

#include <vector>
#include "../linalg/tensor.h"
#include "layer.h"

class SparseLinearInput: public Layer {

	public:
		SparseLinearInput(int, int);

		Tensor* weights;
		Tensor* output;

		void init(int, int);

		void updateOutput(std::vector<int> input);
};

class SparseLinearOutput: public Layer {

	public:
		SparseLinearOutput(int, int, int);

		Tensor* weights;
		Tensor* output;
		std::vector<int> output_indices;

		void init(int, int, int);

		void updateOutput(Tensor* input, std::vector<int> output_indices);
};

#endif