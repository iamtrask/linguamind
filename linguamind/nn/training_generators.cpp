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

CBOW* CBOW::getCopyForSection(int starting, int ending) {
	
	CBOW* new_cbow = new CBOW(this->window_indices, this->vocab, this->sampler, this->window_left, this->window_right, this->model_order);
	
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