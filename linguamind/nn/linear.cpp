#include "linear.h"

Linear::Linear(int input_dim, int output_dim) {
	this->init(input_dim, output_dim);
}

void Linear::init(int input_dim, int output_dim) {

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

Layer* Linear::duplicateWithSameWeights() {

	Linear* new_layer = new Linear(this->input_dim, this->output_dim);

	free(new_layer->weights);
	new_layer->weights = this->weights;

	return (Layer*)new_layer;
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
		this->weights->get(index)->subi(input,output_grad->get(index) * alpha);
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


// TODO: push the algebra from this library to the matrix/vector libraries so that you make fewer method calls.

FlexLinear::FlexLinear(int input_dim, int output_dim) {
	this->init(input_dim, output_dim, true);
}

FlexLinear::FlexLinear(int input_dim, int output_dim, bool init_weights) {
	this->init(input_dim, output_dim, init_weights);
}

// FlexLinear::~FlexLinear() {

// 	// delete this->input;
// 	// delete this->output;

// 	// delete this->input_grad;
// 	// delete this->output_grad;

// 	// this->input_indices.clear();
// 	// this->output_indices.clear();

// 	// delete this->weights;
// }

void FlexLinear::init(int input_dim, int output_dim, bool init_weights) {

	this->output_must_be_sparse = false;
	this->input_must_be_sparse = false;

	this->contains_layers = false;

	this->mandatory_identical_input_output_sparsity = false;

	this->input_dim = input_dim; // embedding dim
	this->output_dim = output_dim; // output vocab

	if(init_weights) {
		this->weights = new Matrix(input_dim, output_dim);
	}

	this->weights_configured_input_sparse = true;

	this->input_grad = new Vector(this->input_dim);
	this->input_grad->zero();

	this->output = new Vector(this->output_dim);
	this->output->zero();

}

int FlexLinear::destroy(bool dont_destroy_weights) {

	this->output->destroy();
	delete this->output;

	this->input_grad->destroy();
	// this->output_grad->destroy();

	delete this->input_grad;
	// delete this->output_grad;

	this->input_indices.clear();
	this->output_indices.clear();
	
	if(!dont_destroy_weights){
		this->weights->destroy();
		delete this->weights;
	}

	return 0;
}

int FlexLinear::swapInputOutputSparsity() {
	
	this->weights->transpose();
	this->weights_configured_input_sparse = (!this->weights_configured_input_sparse);

	return 0;
}

FlexLayer* FlexLinear::duplicateWithSameWeights() {

	FlexLinear* new_layer = new FlexLinear(this->input_dim, this->output_dim,false);

	new_layer->weights = this->weights;
	new_layer->weights_configured_input_sparse = this->weights_configured_input_sparse;

	return (FlexLayer*)new_layer;
}

// dense input, dense output
int FlexLinear::updateOutputDenseToDense(Vector* input) {

	this->forward_code = 0;

	this->input = input;

	if(weights_configured_input_sparse) {

		this->output->set(this->weights->get(0),input->get(0));
		for(int index=1; index < this->input_dim; index++) {
			this->output->addi(this->weights->get(index),input->get(index));
		}

	} else {
		
		for(int index=0; index < this->output_dim; index++) {
			this->output->doti(index, input, this->weights->get(index));
		}

	} 

	return 0;
}

// dense input, weighted sparse output (i.e. word2vec's syn1neg)
int FlexLinear::updateOutputDenseToWeightedSparse(Vector* input, std::vector<int> &output_indices) {

	if(weights_configured_input_sparse) {
		// throw std::runtime_error("ERROR: Flex Linear layer not configured for sparse output.");
		this->swapInputOutputSparsity();
	}

	this->forward_code = 1;

	this->input = input;
	this->output_indices = output_indices;
	
	int output_index;
	for(int i=0; i < this->output_indices.size(); i++) {
		output_index = output_indices[i];
		this->output->doti(output_index, input, this->weights->get(output_index));
	}

	return 0;
}


// weighted sparse input, dense output
int FlexLinear::updateOutputWeightedSparseToDense(Vector* input, std::vector<int> &input_indices) {

	this->forward_code = 2;

	this->input = input;
	this->input_indices = input_indices;

	if(!weights_configured_input_sparse) {
		this->swapInputOutputSparsity();
	}

	int input_indices_size = (int)input_indices.size();

	int input_index;
	if(input_indices_size > 0) {

		this->output->set(this->weights->get(input_indices[0]),input->get(input_indices[0]));
		for(int i=1; i< input_indices_size; i++) {
			input_index = input_indices[i];
			this->output->addi(this->weights->get(input_index), input->get(input_index));
		}

	}

	return 0;
}

// weighted sparse input, weighted sparse output (not sure when i'll ever use this)
int FlexLinear::updateOutputWeightedSparseToWeightedSparse(Vector* input, std::vector<int> &input_indices, std::vector<int> &output_indices) {

	// for now assume output sparse (TODO: ALLOW FOR BOTH)
	if(!weights_configured_input_sparse) {
		this->swapInputOutputSparsity();
	}

	this->forward_code = 3;

	this->input = input;
	this->input_indices = input_indices;
	this->output_indices = output_indices;

	int input_index = 0;
	int output_index = 0;

	for(int i=0; i<output_indices.size(); i++) {
		output_index = output_indices[i];
		this->output->set(output_index, 0);
		for(int j=0; j<input_indices.size(); j++) {
			input_index = input_indices[j];
			this->output->addi(output_index,input->get(input_index) * (this->weights->get(input_index)->get(output_index)));
		}
	}
	return 0;
}

// binary sparse input, dense output (i.e. word2vec's syn0)
int FlexLinear::updateOutputBinarySparseToDense(std::vector<int> &input_indices) {

	this->forward_code = 4;

	this->input_indices = input_indices;

	//ensure input sparsity
	if(!weights_configured_input_sparse) {
		this->swapInputOutputSparsity();
	}

	this->output->set(this->weights->get(input_indices[0]));
	for(int index=1; index < this->input_indices.size(); index++) {
		this->output->addi(this->weights->get(input_indices[index]));
	}

	this->output->divi(input_indices.size()); // normalizes for when you have varying numbers of input indices.

	return 0;
}


// binary sparse input, weighted sparse output (i.e. perceptron)
int FlexLinear::updateOutputBinarySparseToWeightedSparse(std::vector<int> &input_indices, std::vector<int> &output_indices) {

	this->forward_code = 5;

	this->input_indices = input_indices;
	this->output_indices = output_indices;

	//ensure input sparsity
	if(!weights_configured_input_sparse) {
		this->swapInputOutputSparsity();
	}

	for(int i=0; i<this->output_indices.size(); i++) {
		this->output->set(output_indices[i],0);
		for(int j=0; j < input_indices.size(); j++) {
			this->output->addi(output_indices[i], this->weights->get(input_indices[j])->get(output_indices[i]));
		}
	}

	this->output->divi(input_indices.size()); // normalizes for when you have varying numbers of input indices.

	return 0;
}

int FlexLinear::backward(Vector* output_grad) {

	throw std::runtime_error("Error: tried to call backward method on FlexLinear layer");

}

int FlexLinear::updateInputGrad(Vector* output_grad) {
	
	this->output_grad = output_grad;

	// dense to dense
	if(this->forward_code == 0) {
		
		if(this->weights_configured_input_sparse) {
			
			for(int index=0; index < this->input_dim; index++) {
				this->input_grad->doti(index, output_grad, this->weights->get(index));
			}

		} else {

			this->input_grad->set(this->weights->get(0),output_grad->get(0));
			for(int index=1; index < this->output_dim; index++) {
				this->input_grad->addi(this->weights->get(index),output_grad->get(index));
			}

		}

	// dense to weighted sparse
	} else if(this->forward_code == 1) {
		
		this->input_grad->set(this->weights->get(this->output_indices[0]),output_grad->get(this->output_indices[0]));
		for(int index=1; index < this->output_indices.size(); index++) {
			this->input_grad->addi(this->weights->get(this->output_indices[index]),output_grad->get(this->output_indices[index]));
		}	

	// 2 == weighted sparse -> dense
	// 4 == binary sparse -> dense
	} else if(this->forward_code == 2 || this->forward_code == 4) {
		
		for(int index=0; index < this->input_indices.size(); index++) {
			this->input_grad->doti(input_indices[index], output_grad, this->weights->get(input_indices[index]));
		}

	// 3 and 5 are sparse -> sparse
	} else if(this->forward_code == 3 || this->forward_code == 5) {

		// I'll come back to this later... very unlikely to need it for a while.
		throw std::runtime_error("Not Yet Implemented: backprop for sparse to sparse layers.");
		
	} 

	return 0;
}

int FlexLinear::accGradParameters(float alpha) {
	
	// dense to dense
	if(this->forward_code == 0) {

		if(weights_configured_input_sparse) {

			for(int i=0; i<(int)this->input_dim; i++) {
				this->weights->get(i)->subi(output_grad,alpha);
			}

		} else {

			for(int index=0; index < this->output_dim; index++) {
				this->weights->get(index)->subi(this->input,this->output_grad->get(index) * alpha);
			}			

		}

	// dense to weighted sparse
	} else if(this->forward_code == 1) {
		
		int index;
		for(int i=0; i<(int)this->output_indices.size(); i++) {
			index = this->output_indices[i];
			this->weights->get(index)->subi(this->input,this->output_grad->get(index) * alpha);
		}

	// 2 == weighted sparse -> dense
	} else if(this->forward_code == 2) {

		for(int i=0; i<(int)this->input_indices.size(); i++) {
			this->weights->get(this->input_indices[i])->subi(this->output_grad, this->input->get(this->input_indices[i]) * alpha);
		}

	} else if(this->forward_code == 3) {

		// not implemented

	// 4 == binary sparse -> dense
	} else if(this->forward_code == 4) {

		for(int i=0; i<(int)this->input_indices.size(); i++) {
			this->weights->get(this->input_indices[i])->subi(this->output_grad,alpha);
		}

	} else if(this->forward_code == 5) {
		
		// not implemented

	}

	return 0;
}

int FlexLinear::getInputDim() { return this->input_dim;};
int FlexLinear::getOutputDim() { return this->output_dim;};
bool FlexLinear::inputMustBeSparse() {return this->input_must_be_sparse;}
bool FlexLinear::outputMustBeSparse() {return this->output_must_be_sparse;}
bool FlexLinear::containsLayers() {return this->contains_layers;}
Vector* FlexLinear::getOutput() {return this->output;}
std::vector<int> &FlexLinear::getOutputIndices() {return this->output_indices;}
Vector* FlexLinear::getInputGrad() {return this->input_grad;}
int FlexLinear::setOutputGrad(Vector* output_grad) {this->output_grad = output_grad;};
bool FlexLinear::mandatoryIdenticalInputOutputSparsity() {return this->mandatory_identical_input_output_sparsity;}