#include "training_generators.h"

CBOW::CBOW(std::vector<std::vector<int> > &window_indices,Vocab* vocab, int negative, int window) {

	this->vocab = vocab;

	this->window_indices = window_indices;
	this->window_indices_size = this->window_indices.size();
	this->size = this->window_indices_size;

	if(this->window_indices.size() == 0) throw std::runtime_error("ERROR: Cannot initialize CBOW with no training data (rows == 0)");

	this->window = window;
	this->negative = negative;

	this->target_values = new Vector(this->negative+1);
	this->target_values->zero();
	this->target_values->set(0,1);

	this->starting_win_i = 0;

	this->win_i = 0;
	this->pred_i = 0;
	this->window_len = 0;

	this->seed = 0;
	this->has_next = true;

	this->next();

}

CBOW* CBOW::getCopyForSection(int starting, int ending) {
	
	CBOW* new_cbow = new CBOW(this->window_indices, this->vocab, this->negative, this->window);
	
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

	this->seed = this->seed * (unsigned long long)25214903917 + 11; // preserves 1:1 comparison with word2vec

	while(output_indices.size() <= this->negative) {
		this->seed = this->seed * (unsigned long long)25214903917 + 11;
		// printf("\n next random:%llu\n",this->seed);
		// printf("Unigram Index: %i\n",(this->seed >> 16) % this->vocab->unigram_table_size);
		target = this->vocab->unigram_table[(this->seed >> 16) % this->vocab->unigram_table_size];
		// printf("Unigram Value %i\n",target);
		// if (target == 0) target = this->seed % (this->vocab->size - 1) + 1; 
		if (target == 0) continue;
		if (target == word) continue;
		this->output_indices.push_back(target);
	}

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
	return this->target_values;
}