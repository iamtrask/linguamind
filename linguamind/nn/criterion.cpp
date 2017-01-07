#include "criterion.h"

MSECriterion::MSECriterion(int batch_size,int dim) {

	this->batch_size = batch_size;
	this->dim = dim;

	// this->output = new Tensor(batch_size, dim);
	this->grad_input = new Tensor(batch_size, dim);
	
}

void MSECriterion::forwards(Tensor* input) {
	this->output = input;
}

Tensor* MSECriterion::backwards(Tensor* target) {
	this->grad_input->sub(this->output,target);
	return this->grad_input;
}

// void MSECriterion::zeroGradinput()