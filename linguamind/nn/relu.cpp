#include "relu.h"

Relu::Relu(int dim) {

	this->sparse_output = false;
	this->sparse_input = false;

	this->input_dim = dim;
	this->output_dim = dim;

	this->input_grad = new Vector(this->input_dim);
	this->input_grad->zero();

	this->output = new Vector(this->output_dim);
	this->output->zero();

	for(int i=0; i<this->output_dim; i++) this->full_output_indices.push_back(i);

}

int Relu::updateOutput(Vector* input, std::vector<int> output_indices) {

	this->output_indices = output_indices;
	
	this->input_grad->muli((float)0);
	this->input_grad->addi(input);
	this->input_grad->gei((float)0);

	this->output->muli((float)0);
	this->output->addi(input);
	this->output->muli(this->input_grad);

	return 0;
}

int Relu::updateInputGrad(Vector* output_grad) {
	
	this->input_grad->muli(output_grad);

	return 0;
}

int Relu::accGradParameters(Vector* input, Vector* output_grad, float alpha) {
	// do nothing
	return 0;
}

int Relu::getInputDim() { return this->input_dim;};
int Relu::getOutputDim() { return this->output_dim;};
bool Relu::hasSparseInput() {return this->sparse_input;};
bool Relu::hasSparseOutput() {return this->sparse_output;}
Vector* Relu::getOutput() {return this->output;}
Vector* Relu::getInputGrad() {return this->input_grad;}
std::vector<int> Relu::getFullOutputIndices() {return this->full_output_indices;}