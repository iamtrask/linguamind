#include "sigmoid.h"

Sigmoid::Sigmoid(int dim) {

	this->sparse_output = false;
	this->sparse_input = false;

	this->input_dim = dim;
	this->output_dim = dim;

	this->input_grad = new Vector(this->input_dim);
	this->input_grad->zero();

	this->output = new Vector(this->output_dim);
	this->output->zero();

	for(int i=0; i<this->output_dim; i++) this->full_output_indices.push_back(i);

	this->expTable = (float *)malloc((EXP_TABLE_SIZE + 1) * sizeof(float));
  	for (int i = 0; i < EXP_TABLE_SIZE; i++) {
   		this->expTable[i] = exp((i / (float)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
    	this->expTable[i] = this->expTable[i] / (this->expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
  	}

}

Layer* Sigmoid::duplicateWithSameWeights() {
	Sigmoid* new_layer = new Sigmoid(this->input_dim);
	return (Layer*)new_layer;
}

int Sigmoid::updateOutput(Vector* input, std::vector<int> &output_indices) {

	this->output_indices = output_indices;
	
	int len = (int)output_indices.size();
	int index;
	float f;
	for(int i=0; i<len; i++) {
		index = output_indices[i];
		f = input->get(index);
		if (f <= -MAX_EXP) f = 0;
	    else if (f >= MAX_EXP) f = 1;
	    else f = this->expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
		this->output->set(index,f);
		
	}

	return 0;
}

int Sigmoid::updateInputGrad(Vector* output_grad) {
	
	int len = (int)this->output_indices.size();
	float grad;
	for(int i=0; i<len; i++) {
		grad = output_grad->get(output_indices[i]);
		this->input_grad->set(output_indices[i],grad); // word2vec shortcut (no multiply by deriv)
	}

	return 0;
}

int Sigmoid::accGradParameters(Vector* input, Vector* output_grad, float alpha) {
	// do nothing
	return 0;
}

int Sigmoid::getInputDim() { return this->input_dim;};
int Sigmoid::getOutputDim() { return this->output_dim;};
bool Sigmoid::hasSparseInput() {return this->sparse_input;};
bool Sigmoid::hasSparseOutput() {return this->sparse_output;}
Vector* Sigmoid::getOutput() {return this->output;}
Vector* Sigmoid::getInputGrad() {return this->input_grad;}
std::vector<int> Sigmoid::getFullOutputIndices() {return this->full_output_indices;}