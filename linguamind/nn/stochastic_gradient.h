#ifndef SGD
#define SGD

#include <vector>
#include "../linalg/vector.h"
#include "../linalg/matrix.h"
#include "layer.h"
#include "criterion.h"
#include "sequential.h"
#include "training_generators.h"

class StochasticGradient  {

	public:
		StochasticGradient(Sequential* mlp, MSECriterion* criterion);

		Sequential* mlp;
		MSECriterion* criterion;

		float train(CBOW* training_generator, float alpha, int iterations);
};


#endif