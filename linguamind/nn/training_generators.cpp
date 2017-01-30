#include "training_generators.h"

Sampler::Sampler(Vocab* vocab, int sample_size) {
	
	this->vocab = vocab;
	this->seed = 11;

	// if -1 then do a full smple
	this->sample_size = sample_size;


	if(this->sample_size >= 0) {

		this->output_size = this->sample_size+1;

		this->target_values = new Vector(this->output_size);
		this->target_values->zero();
		this->target_values->set(0,1);


	} else {
		this->output_size = this->vocab->size;

		this->target_values = new Vector(this->output_size);
		this->target_values->zero();
		this->target_values->set(0,1);
	}
}

std::vector<int> Sampler::next(std::vector<int> &output_indices) {

	if(output_indices.size() != 1) throw std::runtime_error("ERROR: Sampler expects the output_indices vector to already contain exactly 1 value, the value of the true prediction.");	

	int target;

	while(output_indices.size() < this->output_size) {

		this->seed = this->seed * (unsigned long long)25214903917 + 11;
		target = this->vocab->unigram_table[(this->seed >> 16) % this->vocab->unigram_table_size];

		if (target == 0) continue;
		if (target == output_indices[0]) continue;
		output_indices.push_back(target);

	}

	return output_indices;
}

Vector* Sampler::getTargetValues() {
	return this->target_values;
}

SupervisedBinarySparseToWeightedSparse::SupervisedBinarySparseToWeightedSparse(std::vector<std::vector<int> > input_indices, Vocab* input_vocab, std::vector<std::vector<int> > output_indices, Vocab* output_vocab) {

	this->input_indices = input_indices;
	this->input_vocab = input_vocab;

	this->output_indices = output_indices;
	this->output_vocab = output_vocab;

	if(this->input_indices.size() != this->output_indices.size()) throw std::runtime_error("ERROR: Must have as many labels as you have training data points");

	this->size = this->input_indices.size();

	// by default, 
	this->section_start = 0;
	this->section_end = this->size;

	this->reset();
}

TrainingGenerator* SupervisedBinarySparseToWeightedSparse::getCopyForSection(int starting, int ending) {
	SupervisedBinarySparseToWeightedSparse * gen = new SupervisedBinarySparseToWeightedSparse(this->input_indices, this->input_vocab, this->output_indices, this->output_vocab);

	if(starting < 0 or starting > this->size or ending < 0 or ending > this->size) throw std::runtime_error("ERROR: starting and ending indices must be bewteen 0 and the number of training examples available.");

	gen->section_start = starting;
	gen->section_end = ending;

	return (TrainingGenerator*) gen;
}

int SupervisedBinarySparseToWeightedSparse::reset() {

	this->i = this->section_start;
	this->pred_i = 0;
	this->window_len = this->output_indices[this->i].size();

	this->has_next = true;

}

int SupervisedBinarySparseToWeightedSparse::next() {

	if(this->has_next) {

		this->pred_i += 1;

		if(this->pred_i >= this->window_len) {
			this->i += 1;	
			
			if(this->i >= this->section_end) {
				this->i = this->section_start;
				this->has_next = false;
			}

			this->pred_i = 0;
			this->window_len = this->output_indices[this->i].size();
		}
	}

	return this->has_next;
}

bool SupervisedBinarySparseToWeightedSparse::hasNext() {
	return this->has_next;
}

long SupervisedBinarySparseToWeightedSparse::getSize() {
	return this->size;
}

long SupervisedBinarySparseToWeightedSparse::getI() {
	return this->i;
}

bool SupervisedBinarySparseToWeightedSparse::shouldReset() {
	if(this->pred_i == 0) return true;
	return false;
}

std::vector<int> &SupervisedBinarySparseToWeightedSparse::getInputIndicesReference() {
	return this->input_indices[this->i];
}

std::vector<int> &SupervisedBinarySparseToWeightedSparse::getOutputIndicesReference() {
	return this->output_vocab->getPathReference(this->output_indices[this->i][this->pred_i]);
}
Vector* SupervisedBinarySparseToWeightedSparse::getTargetValuesReference() {
	return this->output_vocab->getCodeReference(this->output_indices[this->i][this->pred_i]);
}



// TODO: create subtraction indices so that you can just tweak the hidden states of a neural network instead of having to regenerate the context window
// for every prediction when only one word is being slotted out and another is being slotted in. With chip caching this might not make much of a difference
// since you still have to store the deltas for each word at each timestep, but it might. 

CBOW::CBOW(std::vector<std::vector<int> > &window_indices,Vocab* vocab,Sampler* sampler, int window_left, int window_right, bool model_order) {

	this->vocab = vocab;
	this->sampler = sampler;
	this->model_order = model_order;

	if(model_order) {
		this->minimum_input_dimensionality = (window_right + window_left) * this->vocab->size;
	} else {
		this->minimum_input_dimensionality = this->vocab->size;
	}

	this->window_indices = window_indices;
	this->window_indices_size = this->window_indices.size();
	this->size = this->window_indices_size;

	if(this->window_indices.size() == 0) throw std::runtime_error("ERROR: Cannot initialize CBOW with no training data (rows == 0)");

	this->window_left = window_left;
	this->window_right = window_right;

	this->starting_win_i = 0;

	this->win_i = 0;
	this->pred_i = 0;
	this->window_len = 0;

	this->seed = 0;
	this->has_next = true;

	this->next();

}

TrainingGenerator* CBOW::getCopyForSection(int starting, int ending) {
	
	CBOW* new_cbow = new CBOW(this->window_indices, this->vocab, this->sampler, this->window_left, this->window_right, this->model_order);
	
	new_cbow->starting_win_i = starting;
	new_cbow->size = ending;

	return (TrainingGenerator*)new_cbow;
}

long CBOW::getSize() {
	return this->size;
}

long CBOW::getI() {
	return this->win_i;
}

bool CBOW::hasNext() {
	return has_next;
}

int CBOW::next() {

	int cur_i;
	long long target = 0;
	long long word;

	if(this->win_i == 0) this->win_i = this->starting_win_i;

	int window_len = (int)this->window_indices[this->win_i].size();
	if(this->pred_i >= window_len) {

		this->pred_i = 0;
		this->win_i++;
		window_len = (int)this->window_indices[this->win_i].size();

		while(window_len < 2 && this->win_i < (int)this->window_indices.size()) {
			this->win_i++;
			window_len = (int)this->window_indices[this->win_i].size();
		}
	}

	if(this->win_i >= (int)this->size) {

		this->win_i = this->starting_win_i;
		this->has_next = false;
		return 0;
	}
	
	this->input_indices.clear();
	this->output_indices.clear();



	if(model_order) {

		// push left (earlier) indices
		for(int cont_i = 0; cont_i < this->window_left; cont_i++) {
			cur_i = pred_i - cont_i - 1;
			if(cur_i >= 0) {
				input_indices.push_back(this->vocab->size * cont_i + this->window_indices[this->win_i][cur_i]);
			}
		}

		// push right (later) indices
		for(int cont_i = 0; cont_i < this->window_right; cont_i++) {
			cur_i = pred_i + cont_i + 1;
			if(cur_i < window_len) {
				input_indices.push_back(this->vocab->size * (cont_i + this->window_left) + this->window_indices[this->win_i][cur_i]);
			}
		}
	} else {
		// push left (earlier) indices
		for(int cont_i = 0; cont_i < this->window_left; cont_i++) {
			cur_i = pred_i - cont_i - 1;
			if(cur_i >= 0) {
				input_indices.push_back(this->window_indices[this->win_i][cur_i]);
			}
		}

		// push right (later) indices
		for(int cont_i = 0; cont_i < this->window_right; cont_i++) {
			cur_i = pred_i + cont_i + 1;
			if(cur_i < window_len) {
				input_indices.push_back(this->window_indices[this->win_i][cur_i]);
			}
		}
	}
	
	word = this->window_indices[this->win_i][pred_i];
	this->output_indices.push_back(word);
	
	this->sampler->next(this->output_indices);

	this->pred_i++;

	if(this->input_indices.size() == 0) this->next();
	return 0;
}

int CBOW::reset() {
	this->has_next = true;
	this->win_i = this->starting_win_i;
	this->pred_i = 0;
	return 0;
}

bool CBOW::shouldReset() {return false;}

std::vector<int> &CBOW::getInputIndicesReference() {
	return this->input_indices;
}

std::vector<int> &CBOW::getOutputIndicesReference() {
	return this->output_indices;
}

Vector* CBOW::getTargetValuesReference() {
	return this->sampler->getTargetValues();
}