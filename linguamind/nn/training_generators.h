#ifndef TRAINING_GENERATOR
#define TRAINING_GENERATOR

#include <vector>
#include "../linalg/vector.h"
#include "../linalg/matrix.h"
#include "layer.h"
#include "criterion.h"
#include "sequential.h"

class CBOW  {

	public:
		CBOW(std::vector<std::vector<int> > window_indices,int vocab_size, int negative, int window);

		std::vector<std::vector<int> > window_indices;
		
		std::vector<int> input_indices;
		std::vector<int> output_indices;

		Vector* target_values;

		int vocab_size;
		int iterator;
		int window;
		int negative;
		int win_i;
		int pred_i;
		int window_len;

		unsigned long long seed;
		bool has_next;

		void next();
		void reset();
		std::vector<int> &getInputIndicesReference();
		std::vector<int> &getOutputIndicesReference();
		Vector* getTargetValuesReference();

};


#endif