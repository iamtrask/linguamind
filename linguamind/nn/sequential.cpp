#include "sequential.h"

#include <memory>

Sequential::Sequential(std::vector<Layer*> layers) {
	for (int i=0; i<layers.size(); i++) {
		this->layers.push_back(layers[i]);
		this->output = layers[i]->getOutput();
	}
}

// void Sequential::add(Layer* layer) {
// 	this->layers.push_back(layer);
	
// }

Layer* Sequential::get(int i) {
	return this->layers[i];
}

// forward with sparse input and output
Vector* Sequential::forward(std::vector<int> input_indices,std::vector<int> output_indices) {
	int num_layers = (int)this->layers.size();
	if(num_layers > 1) {
		this->layers[0]->updateOutput(NULL, input_indices);

		bool sparse_output_until_end = false;

		for(int index=0; index<(int)num_layers-2; index++) {
			if(this->layers[index+1]->hasSparseOutput()) sparse_output_until_end = true;

			if(sparse_output_until_end) {
				this->layers[index+1]->updateOutput(this->layers[index]->getOutput(),output_indices);
			} else {
				this->layers[index+1]->updateOutput(this->layers[index]->getOutput(),this->layers[index+1]->getFullOutputIndices());
			}
		}	

		this->layers[num_layers-1]->updateOutput(this->layers[num_layers-2]->getOutput(),output_indices);
	}
	return this->output;
}

void Sequential::backward(Vector* grad, std::vector<int> output_indices) {
	this->layers[this->layers.size()-1]->updateInputGrad(grad);

	for (int i=this->layers.size()-3; i >= 0; i--) {
		this->layers[i+1]->updateInputGrad(this->layers[i+2]->getInputGrad());
	}
}