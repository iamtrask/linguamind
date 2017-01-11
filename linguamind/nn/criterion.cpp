#include "criterion.h"
#include <stdexcept> 

MSECriterion::MSECriterion() {
	this->grad = new Vector(32);	
}

float MSECriterion::forward(Vector* input, Vector* target, std::vector<int> &output_indices) {
	
	if(target->size != output_indices.size()) {
		printf("Target Size:%i", target->size);
		printf("Output Indices Size:%i", output_indices.size());
		throw std::runtime_error("OutOfBounds: target and output_indices vectors should be of identical length.");
	}

	float error = 0;
	float tmp;
	int index;

	for(int i=0; i<(int)output_indices.size(); i++) {
		index = output_indices[i];
		tmp = (input->get(index) - target->get(i));
		error += (tmp * tmp);
	}

	return error / (float)output_indices.size();
}

Vector* MSECriterion::backward(Vector* output, Vector* target, std::vector<int> &output_indices) {
	
	if(target->size != (int)output_indices.size()) throw std::runtime_error("OutOfBounds: output_indices and target vectors should be of identical length.");

	// this->grad be the size of the whole output... 
	if(this->grad->size != output->size) {
		this->grad = new Vector(output->size);
	}

	int index;
	for(int i=0; i<(int)output_indices.size(); i++) {
		index = output_indices[i];
		this->grad->set(index,output->get(index) - target->get(i));
	}

	return this->grad;
}

