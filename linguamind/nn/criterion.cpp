#include "criterion.h"
#include <stdexcept> 

MSECriterion::MSECriterion() {
	this->grad = new Vector(32);	
}

MSECriterion* MSECriterion::duplicate() {
	return new MSECriterion();
}

void MSECriterion::destroy() {
	this->grad->destroy();
	delete this->grad;
}

float MSECriterion::forward(Vector* input, Vector* target) {


	if(target->size != (int)input->size) throw std::runtime_error("OutOfBounds: input and target vectors should be of identical length.");

	float error = 0;
	float tmp;
	int index;

	for(int i=0; i<target->size; i++) {
		tmp = (input->get(i) - target->get(i));
		error += (tmp * tmp);
	}

	return error / (float)target->size;
}

float MSECriterion::forward(Vector* input, Vector* target, std::vector<int> &output_indices) {

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

Vector* MSECriterion::backward(Vector* output, Vector* target) {
	
	if(target->size != (int)output->size) throw std::runtime_error("OutOfBounds: output and target vectors should be of identical length.");

	// this->grad be the size of the whole output... 
	if(this->grad->size != output->size) {
		this->grad = new Vector(output->size);
	}

	int index;
	for(int i=0; i<target->size; i++) {
		this->grad->set(index,output->get(i) - target->get(i));

	}

	return this->grad;
}

Vector* MSECriterion::backward(Vector* output, Vector* target, std::vector<int> output_indices) {

	// this->grad be the size of the whole output... 
	if(this->grad->size != output->size) {
		this->grad = new Vector(output->size);
	}

	int index;
	for(int i=0; i<(int)output_indices.size(); i++) {
		index = output_indices[i];
		// printf("Target:%f ",target->get(i));
		// printf("Pred:%f\n",output->get(index));
		this->grad->set(index,output->get(index) - target->get(i));

	}

	return this->grad;
}

