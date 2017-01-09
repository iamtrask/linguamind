#ifndef CRITERION
#define CRITERION

#include <vector>
#include "../linalg/vector.h"
#include "../linalg/matrix.h"
#include "layer.h"

class MSECriterion {

	public:	
		MSECriterion();

		Vector* grad;

		float forward(Vector* input, Vector* target, std::vector<int> output_indices);
		Vector* backward(Vector* output, Vector* target, std::vector<int> output_indices);
};

#endif