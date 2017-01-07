#ifndef CRITERION
#define CRITERION

#include <vector>
#include "../linalg/tensor.h"
#include "layer.h"

class MSECriterion {

	public:
		int batch_size;
		int dim;

		Tensor* output;
		Tensor* grad_input;
	
		MSECriterion(int batch_size, int dim);

		void forwards(Tensor* input);
		Tensor* backwards(Tensor* target);
};

#endif