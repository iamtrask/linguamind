#include "linear.h"

Linear::Linear(int input_dim, int output_dim) {

	this->sparse_output = false;
	this->sparse_input = false;

	this->input_dim = input_dim; // embedding dim
	this->output_dim = output_dim; // output vocab

	this->weights = new Matrix(output_dim, input_dim);

	this->input_grad = new Vector(this->input_dim);
	this->input_grad->zero();

	this->output = new Vector(this->output_dim);
	this->output->zero();

	for(int i=0; i<this->output_dim; i++) this->full_output_indices.push_back(i);

}

int Linear::updateOutput(Vector* input, std::vector<int> &not_used) {

	for(int index=0; index < this->output_dim; index++) {
		this->output->doti(index, input, this->weights->get(index));
	}
	return 0;
}

int Linear::updateInputGrad(Vector* output_grad) {
	
	int index = 0;
	this->input_grad->set(this->weights->get(index), output_grad->get(index));
	for(index=1; index < this->output_dim; index++) {
		this->input_grad->addi(this->weights->get(index), output_grad->get(index));
	}
	return 0;
}

int Linear::accGradParameters(Vector* input, Vector* output_grad, float alpha) {
	
	for(int index=0; index < this->output_dim; index++) {
		this->weights->get(index)->addi(input,output_grad->get(index) * -alpha);
	}
	return 0;
}

int Linear::getInputDim() { return this->input_dim;};
int Linear::getOutputDim() { return this->output_dim;};
bool Linear::hasSparseInput() {return this->sparse_input;};
bool Linear::hasSparseOutput() {return this->sparse_output;}
Vector* Linear::getOutput() {return this->output;}
Vector* Linear::getInputGrad() {return this->input_grad;}
std::vector<int> Linear::getFullOutputIndices() {return this->full_output_indices;}