#include "sequential.h"

Sequential::Sequential() {
	
}

void Sequential::add(Layer* layer) {
	this->layers.push_back(layer);
}


// forward with sparse input and output
Tensor* Sequential::forward(std::vector<int> input_indices,std::vector<int> output_indices) {

}