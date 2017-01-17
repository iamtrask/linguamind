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

CBOW::CBOW(std::vector<std::vector<int> > &window_indices,Vocab* vocab,Sampler* sampler, int window) {

	this->vocab = vocab;
	this->sampler = sampler;

	this->window_indices = window_indices;
	this->window_indices_size = this->window_indices.size();
	this->size = this->window_indices_size;

	if(this->window_indices.size() == 0) throw std::runtime_error("ERROR: Cannot initialize CBOW with no training data (rows == 0)");

	this->window = window;

	this->starting_win_i = 0;

	this->win_i = 0;
	this->pred_i = 0;
	this->window_len = 0;

	this->seed = 0;
	this->has_next = true;

	this->next();

}

CBOW* CBOW::getCopyForSection(int starting, int ending) {
	
	CBOW* new_cbow = new CBOW(this->window_indices, this->vocab, this->sampler, this->window);
	
	new_cbow->starting_win_i = starting;
	new_cbow->size = ending;

	return new_cbow;
}

void CBOW::next() {
			
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
		return;
	}
	
	this->input_indices.clear();
	this->output_indices.clear();

	for(int cont_i = 0; cont_i<this->window*2+1; cont_i++) {
		cur_i = pred_i + cont_i - this->window;
		if(cur_i >=0 && cur_i < window_len && cur_i != this->pred_i) {
			input_indices.push_back(this->window_indices[this->win_i][cur_i]);
		}
	}

	word = this->window_indices[this->win_i][pred_i];
	this->output_indices.push_back(word);

	this->sampler->next(this->output_indices);

	this->pred_i++;
}

void CBOW::reset() {
	this->has_next = true;
	this->win_i = this->starting_win_i;
	this->pred_i = 0;
}

std::vector<int> &CBOW::getInputIndicesReference() {
	return this->input_indices;
}

std::vector<int> &CBOW::getOutputIndicesReference() {
	return this->output_indices;
}

Vector* CBOW::getTargetValuesReference() {
	return this->sampler->getTargetValues();
}