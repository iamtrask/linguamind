#include "sequential.h"

Sequential::Sequential() {
	
}

void Sequential::add(Layer* layer) {
	this->layers.push_back(layer);
}


// forward with sparse input and output
Tensor* Sequential::forward(std::vector<int> input_indices,std::vector<int> output_indices) {
	int num_layers = this->layers.size();

	SparseLinearInput* input_layer = (SparseLinearInput*)this->layers[0];

	SparseLinearOutput* output_layer = (SparseLinearOutput*)this->layers[1];

	input_layer->updateOutput(input_indices);
	output_layer->updateOutput(input_layer->output,output_indices);

	this->output = output_layer->output;
	return this->output;
}