#include "sparse_linear.h"

SparseLinearInput::SparseLinearInput(int input_dim, int output_dim) {
	this->init(input_dim, output_dim);
}

void SparseLinearInput::init(int input_dim, int output_dim) {
	
	std::vector<int> weight_dims;

	weight_dims.push_back(input_dim);
	weight_dims.push_back(output_dim);
	
	this->weights = new Tensor(weight_dims);	

	std::vector<int> output_dims;

	output_dims.push_back(1);
	output_dims.push_back(output_dim);
	this->output = new Tensor(output_dims);
}

void SparseLinearInput::updateOutput(std::vector<int> input) {
	
	this->output->zero();

	for(int i=0; i<input.size(); i++) {
		this->output->addRowi(this->weights,input[i]);
	}

}

//////////////////////// SPARSE LINEAR OUTPUT ////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

SparseLinearOutput::SparseLinearOutput(int input_dim, int output_dim, int output_sample_size) {
	this->init(input_dim, output_dim, output_sample_size);
}

void SparseLinearOutput::init(int input_dim, int output_dim, int output_sample_size) {
	
	std::vector<int> weight_dims;
	
	weight_dims.push_back(output_dim);
	weight_dims.push_back(input_dim);
	
	this->weights = new Tensor(weight_dims);

	std::vector<int> output_dims;
	output_dims.push_back(1);
	output_dims.push_back(output_sample_size);	
	this->output = new Tensor(output_dims);
}

void SparseLinearOutput::updateOutput(Tensor* input, std::vector<int> output_indices) {
	
	this->output_indices = output_indices;

	for(int i=0; i<output_indices.size(); i++) {
		this->output->_data[i] = (this->weights->dotRow(input,output_indices[i]));
	}

}
