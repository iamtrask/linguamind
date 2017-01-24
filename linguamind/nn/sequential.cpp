#include "sequential.h"

#include <memory>

// TODO: if people make a sequential with layers that are all copies of the same layer, the gradients get messed up.
// The fix is to check somehow and force them to "duplicateWithSameWeights".

Sequential::Sequential(std::vector<Layer*> layers) {
	for (int i=0; i<layers.size(); i++) {
		this->layers.push_back(layers[i]);
		this->output = layers[i]->getOutput();
	}
}

Layer* Sequential::get(int i) {
	return this->layers[i];
}

Sequential* Sequential::duplicateWithSameWeights() {

	std::vector<Layer*> new_layers;

	for(int i=0; i<this->layers.size(); i++) {
		new_layers.push_back(this->layers[i]->duplicateWithSameWeights());
	}

	Sequential* new_seq = new Sequential(new_layers);
	
	return new_seq;
}

// forward with sparse input and output
Vector* Sequential::forward(std::vector<int> &input_indices,std::vector<int> &output_indices) {
	int num_layers = (int)this->layers.size();
	std::vector<int> fullOutputIndices;
	if(num_layers > 1) {

		this->layers[0]->updateOutput(NULL, input_indices);

		bool sparse_output = false;

		for(int index=0; index<(int)num_layers-2; index++) {

			if(this->layers[index+1]->hasSparseOutput()) sparse_output = true;

			if(sparse_output) {
				this->layers[index+1]->updateOutput(this->layers[index]->getOutput(),output_indices);
			} else {
				fullOutputIndices = this->layers[index+1]->getFullOutputIndices();
				this->layers[index+1]->updateOutput(this->layers[index]->getOutput(),fullOutputIndices);
			}

		}

		this->layers[num_layers-1]->updateOutput(this->layers[num_layers-2]->getOutput(),output_indices);
	}
	return this->output;
}

void Sequential::backward(Vector* grad, std::vector<int> &output_indices) {
	this->layers[this->layers.size()-1]->updateInputGrad(grad);

	for (int i=this->layers.size()-3; i >= 0; i--) {
		this->layers[i+1]->updateInputGrad(this->layers[i+2]->getInputGrad());
	}
}


FlexSequential::FlexSequential(std::vector<FlexLayer*> layers) {
	
	for (int i=0; i<layers.size(); i++) {
		this->layers.push_back(layers[i]);
		this->output = layers[i]->getOutput();
	}

	this->num_layers = this->layers.size();
	this->input_must_be_sparse = this->layers[0]->inputMustBeSparse();
	this->output_must_be_sparse = this->layers[this->num_layers-1]->outputMustBeSparse();

	this->init(layers[0]->getInputDim(),layers[layers.size()-1]->getOutputDim());

	this->layer_index_to_begin_using_sequence_output_indices = layers.size()-1;

	for (int i=0; i<layers.size(); i++) {
		if(this->layers[this->layers.size() - i - 1]->mandatoryIdenticalInputOutputSparsity() == false) {
			break;
		} else {
			this->layer_index_to_begin_using_sequence_output_indices--;
		}
	}

}

int FlexSequential::getLayerIndexToBeginUsingSequenceOutputIndices() {
	return this->layer_index_to_begin_using_sequence_output_indices;
}

void FlexSequential::init(int input_dim, int output_dim) {

	this->output_must_be_sparse = false;
	this->input_must_be_sparse = false;

	this->contains_layers = true;

	this->mandatory_identical_input_output_sparsity = false;

	this->input_dim = input_dim; // embedding dim
	this->output_dim = output_dim; // output vocab

	this->input_grad = new Vector(this->input_dim);
	this->input_grad->zero();

	this->output = new Vector(this->output_dim);
	this->output->zero();

	for(int i=0; i<this->output_dim; i++) this->full_output_indices.push_back(i);

}

Vector* FlexSequential::forward(std::vector<int> &input_indices,std::vector<int> &output_indices) {
	this->updateOutputBinarySparseToWeightedSparse(input_indices, output_indices);
	return this->output;
}

int FlexSequential::backward(Vector* output_grad) {

	// this->updateInputGrad(output_grad);

	for(int i=this->num_layers-1; i >= 1; i--) {	
		if(i==this->num_layers-1) {
			this->layers[i]->updateInputGrad(output_grad);
		} else {
			this->layers[i]->updateInputGrad(this->layers[i+1]->getInputGrad());	
		}
	}

	if(this->layers[0]->containsLayers()) {
		this->layers[0]->backward(this->layers[1]->getInputGrad());
	} else {
		this->layers[0]->setOutputGrad(this->layers[1]->getInputGrad());
	}

	return 0;	
}

FlexLayer* FlexSequential::get(int i) {
	return this->layers[i];
}

FlexLayer* FlexSequential::duplicateWithSameWeights() {

	std::vector<FlexLayer*> new_layers;

	for(int i=0; i<this->layers.size(); i++) {
		new_layers.push_back(this->layers[i]->duplicateWithSameWeights());
	}

	FlexSequential* new_seq = new FlexSequential(new_layers);
	
	return (FlexLayer*)new_seq;
}

FlexSequential* FlexSequential::duplicateSequentialWithSameWeights() {

	std::vector<FlexLayer*> new_layers;

	for(int i=0; i<this->layers.size(); i++) {
		new_layers.push_back(this->layers[i]->duplicateWithSameWeights());
	}

	FlexSequential* new_seq = new FlexSequential(new_layers);
	
	return new_seq;
}


// dense input, dense output
int FlexSequential::updateOutputDenseToDense(Vector* seq_input) {
	
	if(this->input_must_be_sparse || this->output_must_be_sparse) {
		throw std::runtime_error("int FlexSequential::updateOutputDenseToDense: Error: Either your input or output is sparse. Both must be dense for this method.");
	}
	
	Vector* input = seq_input;

	bool prev_layer_sparse = false;
	for(int i=0; i<this->num_layers; i++) {
		if(i < this->num_layers - 1 && (this->layers[i+1]->inputMustBeSparse() || this->layers[i]->outputMustBeSparse())) {
			
			if(prev_layer_sparse) {
				this->layers[i]->updateOutputWeightedSparseToWeightedSparse(input, this->layers[i-1]->getOutputIndices(), this->not_used);	
			} else {
				this->layers[i]->updateOutputDenseToWeightedSparse(input, this->not_used);	
			}

			prev_layer_sparse = true;

		} else {

			if(prev_layer_sparse) {
				this->layers[i]->updateOutputWeightedSparseToDense(input, this->layers[i-1]->getOutputIndices());	
			} else {
				this->layers[i]->updateOutputDenseToDense(input);	
			}

			prev_layer_sparse = false;
		}
		input = this->layers[i]->getOutput();
	}
	
	this->output = input;

	return 0;
}

// dense input, weighted sparse output (i.e. word2vec's syn1neg)
int FlexSequential::updateOutputDenseToWeightedSparse(Vector* seq_input, std::vector<int> &output_indices) {

	if(this->input_must_be_sparse) {
		throw std::runtime_error("int FlexSequential::updateOutputDenseToWeightedSparse: Error: your first layer must have sparse input to call this method.");
	}

	Vector* input = seq_input;

	bool prev_layer_sparse = false;
	int threshold_layer = this->getLayerIndexToBeginUsingSequenceOutputIndices();
	for(int i=0; i<threshold_layer; i++) {
		if((this->layers[i+1]->inputMustBeSparse() || this->layers[i]->outputMustBeSparse())) {
			
			if(prev_layer_sparse) {
				// std::cout << "\n\n1:updateOutputWeightedSparseToWeightedSparse\n\n";
				this->layers[i]->updateOutputWeightedSparseToWeightedSparse(input, this->layers[i-1]->getOutputIndices(), this->not_used);	
			} else {
				// std::cout << "\n\n2:updateOutputDenseToWeightedSparse\n\n";
				this->layers[i]->updateOutputDenseToWeightedSparse(input, this->not_used);	
			}

			prev_layer_sparse = true;

		} else {

			if(prev_layer_sparse) {
				// std::cout << "\n\n3:updateOutputWeightedSparseToDense\n\n";
				this->layers[i]->updateOutputWeightedSparseToDense(input, this->layers[i-1]->getOutputIndices());	
			} else {
				// std::cout << "\n\n4:updateOutputDenseToDense\n\n";
				this->layers[i]->updateOutputDenseToDense(input);	
			}

			prev_layer_sparse = false;
		}
		input = this->layers[i]->getOutput();
	}

	for(int i=threshold_layer; i < this->num_layers; i++) {

		if(prev_layer_sparse) {
			// std::cout << "\n\n5:updateOutputWeightedSparseToWeightedSparse\n\n";
			this->layers[i]->updateOutputWeightedSparseToWeightedSparse(input, this->layers[i-1]->getOutputIndices(), output_indices);
		} else {
			// std::cout << "\n\n6:updateOutputDenseToWeightedSparse\n\n";
			this->layers[i]->updateOutputDenseToWeightedSparse(input, output_indices);
		}
		prev_layer_sparse = true;
		input = this->layers[i]->getOutput();
	}
	
	this->output = this->layers[this->num_layers-1]->getOutput();

	return 0;
}


// weighted sparse input, dense output
int FlexSequential::updateOutputWeightedSparseToDense(Vector* seq_input, std::vector<int> &input_indices) {


	if(this->output_must_be_sparse) {
		throw std::runtime_error("int FlexSequential::updateOutputWeightedSparseToDense: Error: Output must be dense. Your last layer requires dense output.");
	}
	
	Vector* input = seq_input;

	bool prev_layer_sparse = true;
	for(int i=0; i<this->num_layers; i++) {
		if(i < this->num_layers - 1 && (this->layers[i+1]->inputMustBeSparse() || this->layers[i]->outputMustBeSparse())) {
			
			if(prev_layer_sparse) {

				if(i==0) {
					this->layers[i]->updateOutputWeightedSparseToWeightedSparse(input, this->layers[i-1]->getOutputIndices(), input_indices);
				} else {
					this->layers[i]->updateOutputWeightedSparseToWeightedSparse(input, this->layers[i-1]->getOutputIndices(), this->not_used);		
				}
				
			} else {
				this->layers[i]->updateOutputDenseToWeightedSparse(input, this->not_used);	
			}

			prev_layer_sparse = true;

		} else {

			if(prev_layer_sparse) {
				if(i==0) {
					this->layers[i]->updateOutputWeightedSparseToDense(input, input_indices);	
				} else {
					this->layers[i]->updateOutputWeightedSparseToDense(input, this->layers[i-1]->getOutputIndices());		
				}
				
			} else {
				this->layers[i]->updateOutputDenseToDense(input);	
			}

			prev_layer_sparse = false;
		}
		input = this->layers[i]->getOutput();
	}
	
	this->output = input;

	return 0;
}

// weighted sparse input, weighted sparse output (not sure when i'll ever use this)
int FlexSequential::updateOutputWeightedSparseToWeightedSparse(Vector* seq_input, std::vector<int> &input_indices, std::vector<int> &output_indices) {
	
	Vector* input = seq_input;

	FlexLayer* cur_layer;
	FlexLayer* prev_layer;

	bool prev_layer_sparse = true;
	int threshold_layer = this->getLayerIndexToBeginUsingSequenceOutputIndices();
	for(int i=0; i<threshold_layer; i++) {
		cur_layer = this->layers[i];
		if(i < this->num_layers - 1 && (this->layers[i+1]->inputMustBeSparse() || cur_layer->outputMustBeSparse())) {
			
			if(prev_layer_sparse) {

				if(i==0) {
					cur_layer->updateOutputWeightedSparseToWeightedSparse(input, prev_layer->getOutputIndices(), input_indices);
				} else {
					cur_layer->updateOutputWeightedSparseToWeightedSparse(input, prev_layer->getOutputIndices(), this->not_used);		
				}
				
			} else {
				cur_layer->updateOutputDenseToWeightedSparse(input, this->not_used);	
			}

			prev_layer_sparse = true;

		} else {

			if(prev_layer_sparse) {
				if(i==0) {
					cur_layer->updateOutputWeightedSparseToDense(input, input_indices);	
				} else {
					cur_layer->updateOutputWeightedSparseToDense(input, prev_layer->getOutputIndices());		
				}
				
			} else {
				cur_layer->updateOutputDenseToDense(input);	
			}

			prev_layer_sparse = false;
		}
		input = cur_layer->getOutput();

		prev_layer = cur_layer;
	}
	
	for(int i=threshold_layer; i < this->num_layers; i++) {

		if(prev_layer_sparse) {
			// std::cout << "\n\n5:updateOutputWeightedSparseToWeightedSparse\n\n";
			this->layers[i]->updateOutputWeightedSparseToWeightedSparse(input, this->layers[i-1]->getOutputIndices(), output_indices);
		} else {
			// std::cout << "\n\n6:updateOutputDenseToWeightedSparse\n\n";
			this->layers[i]->updateOutputDenseToWeightedSparse(input, output_indices);
		}
		prev_layer_sparse = true;
		input = this->layers[i]->getOutput();
	}

	this->output = this->layers[this->num_layers-1]->getOutput();

	return 0;
}

// binary sparse input, dense output (i.e. word2vec's syn0)
int FlexSequential::updateOutputBinarySparseToDense(std::vector<int> &input_indices) {

	if(this->output_must_be_sparse) {
		throw std::runtime_error("int FlexSequential::updateOutputBinarySparseToDense: Error: Output is sparse. Try binarySpareToWeightedSparse()");
	}
	

	bool prev_layer_sparse = false;

	if((this->layers[1]->inputMustBeSparse() || this->layers[0]->outputMustBeSparse())) {

		this->layers[0]->updateOutputBinarySparseToWeightedSparse(input_indices, this->not_used);	
		prev_layer_sparse = true;
	} else {

		this->layers[0]->updateOutputBinarySparseToDense(input_indices);			
		// prev_layer_sparse = false; // redundant
	}

	Vector* input = this->layers[0]->getOutput();

	for(int i=1; i<this->num_layers; i++) {

		if(i < this->num_layers - 1 && (this->layers[i+1]->inputMustBeSparse() || this->layers[i]->outputMustBeSparse())) {
			
			if(prev_layer_sparse) {
				this->layers[i]->updateOutputWeightedSparseToWeightedSparse(input, this->layers[i-1]->getOutputIndices(), this->not_used);	
			} else {
				this->layers[i]->updateOutputDenseToWeightedSparse(input, this->not_used);	
			}	

			prev_layer_sparse = true;

		} else {

			if(prev_layer_sparse) {
				this->layers[i]->updateOutputWeightedSparseToDense(input, this->layers[i-1]->getOutputIndices());	
			} else {
				this->layers[i]->updateOutputDenseToDense(input);	
			}

			prev_layer_sparse = false;
		}
		input = this->layers[i]->getOutput();
	}
	
	this->output = input;

	return 0;
}


// binary sparse input, weighted sparse output (i.e. perceptron or word2vec as a whole)
int FlexSequential::updateOutputBinarySparseToWeightedSparse(std::vector<int> &input_indices, std::vector<int> &output_indices) {

	bool prev_layer_sparse = false;


	if(this->num_layers == 1) {
		this->layers[0]->updateOutputBinarySparseToWeightedSparse(input_indices, output_indices);
	} else {

		if((this->layers[1]->inputMustBeSparse() || this->layers[0]->outputMustBeSparse())) {

			this->layers[0]->updateOutputBinarySparseToWeightedSparse(input_indices, this->not_used);	
			prev_layer_sparse = true;
		} else {

			this->layers[0]->updateOutputBinarySparseToDense(input_indices);			
			// prev_layer_sparse = false; // redundant
		}


		Vector* input = this->layers[0]->getOutput();
		
		int threshold_layer = this->getLayerIndexToBeginUsingSequenceOutputIndices();

		for(int i=1; i<threshold_layer; i++) {

			if(i < this->num_layers - 1 && (this->layers[i+1]->inputMustBeSparse() || this->layers[i]->outputMustBeSparse())) {
				
				if(prev_layer_sparse) {
					this->layers[i]->updateOutputWeightedSparseToWeightedSparse(input, this->layers[i-1]->getOutputIndices(), this->not_used);	
				} else {
					this->layers[i]->updateOutputDenseToWeightedSparse(input, this->not_used);	
				}	

				prev_layer_sparse = true;

			} else {

				if(prev_layer_sparse) {
					this->layers[i]->updateOutputWeightedSparseToDense(input, this->layers[i-1]->getOutputIndices());	
				} else {
					this->layers[i]->updateOutputDenseToDense(input);	
				}

				prev_layer_sparse = false;
			}
			input = this->layers[i]->getOutput();
		}

		for(int i=threshold_layer; i < this->num_layers; i++) {

			if(prev_layer_sparse) {
				// std::cout << "\n\n5:updateOutputWeightedSparseToWeightedSparse\n\n";
				this->layers[i]->updateOutputWeightedSparseToWeightedSparse(input, this->layers[i-1]->getOutputIndices(), output_indices);
			} else {
				// std::cout << "\n\n6:updateOutputDenseToWeightedSparse\n\n";
				this->layers[i]->updateOutputDenseToWeightedSparse(input, output_indices);
			}
			prev_layer_sparse = true;
			input = this->layers[i]->getOutput();
		}
		
	} 

	this->output = this->layers[this->num_layers-1]->getOutput();
	
	return 0;
}

int FlexSequential::updateInputGrad(Vector* output_grad) {

	for(int i=this->num_layers-1; i >= 0; i--) {	
		if(i==this->num_layers-1) {
			this->layers[i]->updateInputGrad(output_grad);
		} else {
			this->layers[i]->updateInputGrad(this->layers[i+1]->getInputGrad());	
		}
	}

	this->input_grad = this->layers[0]->getInputGrad();

	return 0;
}

int FlexSequential::accGradParameters(float alpha) {

	for(int i=0; i<this->num_layers; i++) {
		this->layers[i]->accGradParameters(alpha);
	}

	return 0;
}

int FlexSequential::getInputDim() { return this->input_dim;};
int FlexSequential::getOutputDim() { return this->output_dim;};
bool FlexSequential::inputMustBeSparse() {return this->input_must_be_sparse;};
bool FlexSequential::outputMustBeSparse() {return this->output_must_be_sparse;};
bool FlexSequential::containsLayers() {return this->contains_layers;}
Vector* FlexSequential::getOutput() {return this->output;}
std::vector<int> &FlexSequential::getOutputIndices() {return this->output_indices;}
Vector* FlexSequential::getInputGrad() {return this->input_grad;}
int FlexSequential::setOutputGrad(Vector* output_grad) {this->output_grad = output_grad;};
std::vector<int> FlexSequential::getFullOutputIndices() {return this->full_output_indices;}
bool FlexSequential::mandatoryIdenticalInputOutputSparsity() {return this->mandatory_identical_input_output_sparsity;}

