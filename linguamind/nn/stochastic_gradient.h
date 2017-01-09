#ifndef SGD
#define SGD

#include <vector>
#include "../linalg/vector.h"
#include "../linalg/matrix.h"
#include "layer.h"
#include "criterion.h"
#include "sequential.h"

class StochasticGradient  {

	public:
		StochasticGradient(Sequential* mlp, MSECriterion* criterion);

		Sequential* mlp;
		MSECriterion* criterion;

		float train(std::vector<int> input_indices, std::vector<int> output_indices, Vector* target_values, float alpha, int iterations);		
};


#endif