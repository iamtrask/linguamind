#include "training_generators.h"

CBOW::CBOW(std::vector<std::vector<int> > window_indices,int vocab_size, int negative, int window) {

	this->vocab_size = 0;

	this->window_indices = window_indices;
	this->window = window;
	this->negative = negative;

	this->target_values = new Vector(this->negative + 1);
	this->target_values->zero();
	this->target_values[0] = 1;

	this->win_i = 0;
	this->pred_i = 0;
	this->window_len = 0;

	this->seed = 0;
	this->has_next = true;

}

void CBOW::next() {
			
	int cur_i;
	// prepare training example
	int window_len = (int)window_indices[this->win_i].size();
		
	this->input_indices.clear();
	this->output_indices.clear();

	for(int cont_i = 0; cont_i<window_len*2+1; cont_i++) {
		cur_i = pred_i + cont_i - this->window;
		if(cur_i >=0 && cur_i < window_len && cur_i != this->pred_i) {
			input_indices.push_back(this->window_indices[this->win_i][cur_i]);
		}
	}

	this->output_indices.push_back(this->window_indices[this->win_i][pred_i]);

	for(int i=0; i<5; i++) {
		this->seed = this->seed * (unsigned long long)25214903917 + 11;
		this->output_indices.push_back((int)(this->seed % 29471));
	}

	this->pred_i++;

	if(this->pred_i >= window_len) {

		this->win_i++;
		this->pred_i = 0;

		if(this->win_i >= (int)this->window_indices.size()) {

			this->win_i = 0;
			this->has_next = false;

		}
	}	

}

void CBOW::reset() {
	this->has_next = true;
	this->win_i = 0;
	this->pred_i = 0;
}

std::vector<int> &CBOW::getInputIndicesReference() {
	return this->input_indices;
}

std::vector<int> &CBOW::getOutputIndicesReference() {
	return this->output_indices;
}

Vector* CBOW::getTargetValuesReference() {
	return this->target_values;
}