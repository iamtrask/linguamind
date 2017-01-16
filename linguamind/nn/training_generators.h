#ifndef TRAINING_GENERATOR
#define TRAINING_GENERATOR

#include <vector>
#include "../linalg/vector.h"
#include "../linalg/matrix.h"
#include "../nlp/vocab.h"
#include "layer.h"
#include "criterion.h"
#include "sequential.h"

class CBOW  {

	public:
		CBOW(std::vector<std::vector<int> > &window_indices,Vocab* vocab, int negative, int window);

		std::vector<std::vector<int> > window_indices;
		long window_indices_size;
		long size;

		std::vector<int> input_indices;
		std::vector<int> output_indices;

		Vector* target_values;
		Vocab* vocab;
		
		int iterator;
		int window;
		int negative;

		int starting_win_i;
		int win_i;
		int pred_i;
		int window_len;

		unsigned long long seed;
		bool has_next;

		void next();
		void reset();
		CBOW* getCopyForSection(int starting, int ending);
		std::vector<int> &getInputIndicesReference();
		std::vector<int> &getOutputIndicesReference();
		Vector* getTargetValuesReference();

};


#endif