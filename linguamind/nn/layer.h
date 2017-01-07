#ifndef LAYER
#define LAYER

#include "../linalg/seed.h"
#include "../linalg/vector.h"
#include "../linalg/matrix.h"

class Layer {

	public:
		bool sparse_output;
		bool sparse_input;

		int input_dim;
		int output_dim;

		Vector* output;
		Vector* input_grad;

		std::vector<int> full_output_indices;
};
#endif