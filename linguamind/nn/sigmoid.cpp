#include "sigmoid.h"

Sigmoid::Sigmoid(int dim) {
	this->init(dim);
}

void Sigmoid::init(int dim) {
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


FlexSigmoid::FlexSigmoid(int dim) {


	this->output_must_be_sparse = false;
	this->input_must_be_sparse = false;

	this->contains_layers = false;

	this->mandatory_identical_input_output_sparsity = true;

	this->input_dim = dim; // embedding dim
	this->output_dim = dim; // output vocab

	this->input_grad = new Vector(this->input_dim);
	this->input_grad->zero();

	this->output = new Vector(this->output_dim);
	this->output->zero();

	// for(int i=0; i<this->output_dim; i++) this->full_output_indices.push_back(i);

	this->expTable = (float *)malloc((EXP_TABLE_SIZE + 1) * sizeof(float));
	for (int i = 0; i < EXP_TABLE_SIZE; i++) {
   		this->expTable[i] = exp((i / (float)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
    	this->expTable[i] = this->expTable[i] / (this->expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
    }

}

FlexLayer* FlexSigmoid::duplicateWithSameWeights() {
	FlexSigmoid* new_layer = new FlexSigmoid(this->input_dim);
	return (FlexLayer*)new_layer;
}

int FlexSigmoid::updateOutputDenseToDense(Vector* input) {

	this->forward_code = 0;

	int index;
	float f;
	for(int index=0; index<this->input_dim; index++) {
		
		f = input->get(index);
		if (f <= -MAX_EXP) f = 0;
		else if (f >= MAX_EXP) f = 1;
		else f = this->expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
		this->output->set(index,f);

	}

}

int FlexSigmoid::updateOutputDenseToWeightedSparse(Vector* input, std::vector<int> &sparse_output) {

	this->forward_code = 1;

	throw std::runtime_error("ERROR: Dense to weighted sparse on non-linearity would lose information, deeming the previous dense later a waste. Change previous layer to have sparse output instead.");

}

int FlexSigmoid::updateOutputWeightedSparseToDense(Vector* input, std::vector<int> &input_indices) {

	this->forward_code = 2;

	throw std::runtime_error("ERROR: Weighted sprase doesn't have enough input to fill a dense layer of the identical size. Try weighted sparse to weighted sparse or dense to dense.");

}

int FlexSigmoid::updateOutputWeightedSparseToWeightedSparse(Vector* input, std::vector<int> &input_indices, std::vector<int> &output_indices) {

	this->forward_code = 3;	
	// note, assumes that input_indices and output_indices are identical.
	this->input_indices = input_indices;
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
}

int FlexSigmoid::updateOutputBinarySparseToDense(std::vector<int> &input_indices) {

	this->forward_code = 4;

	throw std::runtime_error("ERROR: Why call signal on binary input?");

}

int FlexSigmoid::updateOutputBinarySparseToWeightedSparse(std::vector<int> &input_indices, std::vector<int> &output_indices) {

	this->forward_code = 5;

	throw std::runtime_error("ERROR: Why call sigmoid on binary input?");

}

int FlexSigmoid::backward(Vector* output_grad) {

	throw std::runtime_error("ERROR: Backword only for layers with sub-layers.");

}

int FlexSigmoid::updateInputGrad(Vector* output_grad) {
	
	float grad;
	if(this->forward_code == 0) {
		
		for(int i=0; i<this->output_dim; i++) {
			grad = output_grad->get(i);
			this->input_grad->set(i,grad);
		}

	} else if (this->forward_code == 3) {

		int len = (int)this->output_indices.size();

		for(int i=0; i<len; i++) {
			grad = output_grad->get(output_indices[i]);
			this->input_grad->set(output_indices[i],grad); // word2vec shortcut (no multiply by deriv)
		}

	}

	return 0;
}

int FlexSigmoid::accGradParameters(float alpha) {
	// do nothing
	return 0;
}

int FlexSigmoid::getInputDim() { return this->input_dim;};
int FlexSigmoid::getOutputDim() { return this->output_dim;};
bool FlexSigmoid::inputMustBeSparse() {return this->input_must_be_sparse;}
bool FlexSigmoid::outputMustBeSparse() {return this->output_must_be_sparse;}
bool FlexSigmoid::containsLayers() {return this->contains_layers;}
Vector* FlexSigmoid::getOutput() {return this->output;}
std::vector<int> &FlexSigmoid::getOutputIndices() {return this->output_indices;}
Vector* FlexSigmoid::getInputGrad() {return this->input_grad;}
int FlexSigmoid::setOutputGrad(Vector* output_grad) {this->output_grad = output_grad;};
std::vector<int> FlexSigmoid::getFullOutputIndices() {return this->full_output_indices;}
bool FlexSigmoid::mandatoryIdenticalInputOutputSparsity() {return this->mandatory_identical_input_output_sparsity;}