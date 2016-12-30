#include "sparse_linear.h"

SparseLinear::SparseLinear(int input_dim, int output_dim) {
	this->init(input_dim, output_dim, false);
}

SparseLinear::SparseLinear(int input_dim, int output_dim, bool is_output_sparse) {
	this->init(input_dim, output_dim, is_output_sparse);
}

void SparseLinear::init(int input_dim, int output_dim, bool is_output_sparse) {
	
	std::vector<int> dims;
	this->is_output_sparse = is_output_sparse;

	if(is_output_sparse) {
		dims.push_back(output_dim);
		dims.push_back(input_dim);
	} else {
		dims.push_back(input_dim);
		dims.push_back(output_dim);
	}

	this->weights = new Tensor(dims);	
}

