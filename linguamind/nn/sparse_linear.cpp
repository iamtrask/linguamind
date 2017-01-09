#include "sparse_linear.h"

SparseLinearInput::SparseLinearInput(int input_dim, int output_dim) {

	this->sparse_output = false;
	this->sparse_input = true;

	this->input_dim = input_dim;
	this->output_dim = output_dim;

	this->weights = new Matrix(input_dim, output_dim);

	this->output = new Vector(this->output_dim);
	this->output->zero();

	for(int i=0; i<this->output_dim; i++) this->full_output_indices.push_back(i);

}

int SparseLinearInput::updateOutput(Vector* input, std::vector<int> input_indices) {
	this->input_indices = input_indices;

	this->output->zero();
	for(int i=0; i< (int)this->input_indices.size(); i++) {
		this->output->addi(this->weights->get(input_indices[i]));
	}
	return 0;
}

int SparseLinearInput::updateInputGrad(Vector* output_grad) {
	// do nothing
	return 0;
}

int SparseLinearInput::accGradParameters(Vector* input, Vector* output_grad, float alpha) {
	for(int i=0; i<(int)this->input_indices.size(); i++) {
		this->weights->get(this->input_indices[i])->addi(output_grad,-alpha);
	}
	return 0;
}

int SparseLinearInput::getInputDim() { return this->input_dim;};
int SparseLinearInput::getOutputDim() { return this->output_dim;};
bool SparseLinearInput::hasSparseInput() {return this->sparse_input;};
bool SparseLinearInput::hasSparseOutput() {return this->sparse_output;}
Vector* SparseLinearInput::getOutput() {return this->output;}
Vector* SparseLinearInput::getInputGrad() {return this->input_grad;}
std::vector<int> SparseLinearInput::getFullOutputIndices() {return this->full_output_indices;}

SparseLinearOutput::SparseLinearOutput(int input_dim, int output_dim) {

	this->sparse_output = true;
	this->sparse_input = false;

	this->input_dim = input_dim; // embedding dim
	this->output_dim = output_dim; // output vocab

	this->weights = new Matrix(output_dim, input_dim);

	this->input_grad = new Vector(this->input_dim);

	this->output = new Vector(this->output_dim);
	this->output->zero();

	for(int i=0; i<this->output_dim; i++) this->full_output_indices.push_back(i);

}

int SparseLinearOutput::updateOutput(Vector* input, std::vector<int> output_indices) {
	this->output_indices = output_indices;
	
	int index = 0;
	for(int i=0; i< (int)this->output_indices.size(); i++) {
		index = this->output_indices[i];
		this->output->doti(index, input, this->weights->get(index));
	}
	return 0;
}

int SparseLinearOutput::updateInputGrad(Vector* output_grad) {
	
	int index = this->output_indices[0];
	this->input_grad->set(this->weights->get(index), output_grad->get(index));
	for(int i=1; i < (int) this->output_indices.size(); i++) {
		index = this->output_indices[i];
		this->input_grad->addi(this->weights->get(index), output_grad->get(index));
	}
	return 0;
}

int SparseLinearOutput::accGradParameters(Vector* input, Vector* output_grad, float alpha) {
	int index;
	for(int i=0; i<(int)this->output_indices.size(); i++) {
		index = this->output_indices[i];
		this->weights->get(index)->addi(input,output_grad->get(index) * -alpha);
	}
	return 0;
}

int SparseLinearOutput::getInputDim() { return this->input_dim;};
int SparseLinearOutput::getOutputDim() { return this->output_dim;};
bool SparseLinearOutput::hasSparseInput() {return this->sparse_input;};
bool SparseLinearOutput::hasSparseOutput() {return this->sparse_output;}
Vector* SparseLinearOutput::getOutput() {return this->output;}
Vector* SparseLinearOutput::getInputGrad() {return this->input_grad;}
std::vector<int> SparseLinearOutput::getFullOutputIndices() {return this->full_output_indices;}