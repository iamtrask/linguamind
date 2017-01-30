#ifndef SGD
#define SGD

#include <vector>
#include "../linalg/vector.h"
#include "../linalg/matrix.h"
#include "layer.h"
#include "criterion.h"
#include "sequential.h"
#include "training_generators.h"

class StochasticGradientWorker{ 

	public:
		StochasticGradientWorker(FlexSequential* mlp, MSECriterion* criterion, TrainingGenerator* training_generator,float alpha, int iterations, int worker_id, int num_workers);

		FlexSequential* mlp;
		MSECriterion* criterion;
		TrainingGenerator* training_generator;

		float alpha;
		int iterations;

		int worker_id;
		int num_workers;

		void train();
		void destroy(bool dont_destroy_weights);

};

class StochasticGradient  {

	public:
		StochasticGradient(FlexSequential* mlp, MSECriterion* criterion);

		FlexSequential* mlp;
		MSECriterion* criterion;
		std::vector<StochasticGradientWorker*> workers;


		float train(TrainingGenerator* training_generator, float alpha, int iterations, int num_threads);
};

void *TrainModelThread(void *sgd);
#endif