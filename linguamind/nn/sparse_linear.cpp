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

Layer* SparseLinearInput::duplicateWithSameWeights() {
	SparseLinearInput* new_layer = new SparseLinearInput(this->input_dim, this->output_dim);
	
	free(new_layer->weights);
	new_layer->weights = this->weights;
	return (Layer*)new_layer;
}

int SparseLinearInput::updateOutput(Vector* input, std::vector<int> &input_indices) {
	this->input_indices = input_indices;

	this->output->zero();
	for(int i=0; i< (int)this->input_indices.size(); i++) {
		this->output->addi(this->weights->get(input_indices[i]));
	}

	this->output->divi((int)this->input_indices.size());
	return 0;
}

int SparseLinearInput::updateInputGrad(Vector* output_grad) {
	// do nothing
	return 0;
}

int SparseLinearInput::accGradParameters(Vector* input, Vector* output_grad, float alpha) {
	for(int i=0; i<(int)this->input_indices.size(); i++) {
		this->weights->get(this->input_indices[i])->subi(output_grad,alpha);
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

Layer* SparseLinearOutput::duplicateWithSameWeights() {
	SparseLinearOutput* new_layer = new SparseLinearOutput(this->input_dim, this->output_dim);
	
	free(new_layer->weights);
	new_layer->weights = this->weights;
	return (Layer*)new_layer;
}

int SparseLinearOutput::updateOutput(Vector* input, std::vector<int> &output_indices) {
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
		this->weights->get(index)->subi(input,output_grad->get(index) * alpha);
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


// NegativeSamplingOutput::NegativeSamplingOutput(int input_dim, int negative_sample_size, int vocab_size) {

// 	this->sparse_output = true;
// 	this->sparse_input = false;

// 	this->input_dim = input_dim; // embedding dim
// 	this->output_dim = vocab_size; // output vocab

// 	this->weights = new Matrix(output_dim, input_dim);

// 	this->input_grad = new Vector(this->input_dim);

// 	this->output = new Vector(this->output_dim);
// 	this->output->zero();

// 	for(int i=0; i<this->output_dim; i++) this->full_output_indices.push_back(i);

// 	this->negative_sample_size = negative_sample_size;
// 	this->vocab_size = vocab_size;

// 	for(int i=0; i<this->negative_sample_size+1; i++) {
// 		this->output_indices.push_back(i);
// 	}

// 	// unsigned long long
// 	this->neg_pos = 1;
	
// }

// int NegativeSamplingOutput::updateOutput(Vector* input, std::vector<int> &output_indices) {
// 	this->output_indices[0] = output_indices[0];
	
// 	int target = this->output_indices[0];
// 	this->output->doti(target, input, this->weights->get(target));
// 	int neg;
// 	for(int i=0; i<this->negative_sample_size; i++) {
// 		neg = this->getRandom(target);
// 		this->output_indices[i+1] = neg;
// 		this->output->doti(neg, input, this->weights->get(neg));
// 	}

// 	return 0;
// }

// int NegativeSamplingOutput::getRandom(int target) {
// 	// TODO fix this to fetch the right distribution
// 	this->neg_pos = this->neg_pos * (unsigned long long)25214903917 + 11;
// 	int output = (int)(this->neg_pos % vocab_size);
// 	while(output == target) {
// 		this->neg_pos = this->neg_pos * (unsigned long long)25214903917 + 11;
// 		output = (int)(this->neg_pos % vocab_size);
// 	}
// 	return output;
// }
// // these are exactly the same as SparseLinearOutput
// int NegativeSamplingOutput::updateInputGrad(Vector* output_grad) {
	
// 	int index = this->output_indices[0];
// 	this->input_grad->set(this->weights->get(index), output_grad->get(index));
// 	for(int i=1; i < (int) this->output_indices.size(); i++) {
// 		index = this->output_indices[i];
// 		this->input_grad->addi(this->weights->get(index), output_grad->get(index));
// 	}
// 	return 0;
// }

// // these are exactly the same as SparseLinearOutput
// int NegativeSamplingOutput::accGradParameters(Vector* input, Vector* output_grad, float alpha) {
// 	int index;
// 	for(int i=0; i<(int)this->output_indices.size(); i++) {
// 		index = this->output_indices[i];
// 		this->weights->get(index)->addi(input,output_grad->get(index) * -alpha);
// 	}
// 	return 0;
// }

// int NegativeSamplingOutput::getInputDim() { return this->input_dim;};
// int NegativeSamplingOutput::getOutputDim() { return this->output_dim;};
// bool NegativeSamplingOutput::hasSparseInput() {return this->sparse_input;};
// bool NegativeSamplingOutput::hasSparseOutput() {return this->sparse_output;}
// Vector* NegativeSamplingOutput::getOutput() {return this->output;}
// Vector* NegativeSamplingOutput::getInputGrad() {return this->input_grad;}
// std::vector<int> NegativeSamplingOutput::getFullOutputIndices() {return this->full_output_indices;}