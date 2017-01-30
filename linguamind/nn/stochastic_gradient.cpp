#include "stochastic_gradient.h"

StochasticGradientWorker::StochasticGradientWorker(FlexSequential* mlp, MSECriterion* criterion, TrainingGenerator* training_generator,float alpha, int iterations, int worker_id, int num_workers) {
	
	this->alpha = alpha;
	this->iterations = iterations;

	this->worker_id = worker_id;
	this->num_workers = num_workers;

	if(this->worker_id != 0) {
		this->mlp = mlp->duplicateSequentialWithSameWeights();
		this->criterion = criterion->duplicate();
	} else {
		this->mlp = mlp;
		this->criterion = criterion;
	}

	long batch_size_per_thread = training_generator->getSize() / (long)num_workers;

	if(worker_id != num_workers - 1) {
		this->training_generator = training_generator->getCopyForSection(worker_id * batch_size_per_thread,(worker_id+1)*batch_size_per_thread);
	} else {
		this->training_generator = training_generator->getCopyForSection(worker_id * batch_size_per_thread,training_generator->getSize());
	}

}

void StochasticGradientWorker::train() {
	
	std::vector<int> &input_indices = this->training_generator->getInputIndicesReference();
	std::vector<int> &output_indices = this->training_generator->getOutputIndicesReference();
	Vector* target_values = this->training_generator->getTargetValuesReference();

	for(int iter=0; iter<this->iterations; iter++) {
		
		while(this->training_generator->hasNext()) {
		
			this->mlp->backward(this->criterion->backward(this->mlp->forward(input_indices,output_indices),target_values,output_indices));

			for(int i=0; i<(int)this->mlp->layers.size(); i++) {
				this->mlp->get(i)->accGradParameters(this->alpha);
			}

			this->training_generator->next();
			if(this->training_generator->shouldReset()) this->mlp->reset(); // run this after next() so that the reset happens before training
		}

		this->training_generator->reset();
	}
}

void StochasticGradientWorker::destroy(bool dont_destroy_weights) {
	if(this->worker_id != 0) {
		this->mlp->destroy(dont_destroy_weights);
		this->criterion->destroy();	
		delete this->mlp;
		delete this->criterion;
	}
	
	delete this->training_generator;
	// TODO: optimize this method... cache mlp and criterion instead of destroying them
	// or just cache the entire SGDWorker...  (although this is relatively minor ... 0.1 ms probably)
	
}

StochasticGradient::StochasticGradient(FlexSequential* mlp, MSECriterion* criterion) {
	this->mlp = mlp;
	this->criterion = criterion;
}

void *TrainModelThread(void *sgd_ptr) {

	StochasticGradientWorker* sgd_worker = ((StochasticGradientWorker *)sgd_ptr);
	sgd_worker->train();

	pthread_exit(NULL);
}

float StochasticGradient::train(TrainingGenerator* training_generator, float alpha, int iterations, int num_threads) {

	// if(training_generator->minimum_input_dimensionality > this->mlp->get(0)->getInputDim()) throw std::runtime_error("ERROR: input layer too small for training example generator. Are you modeling order? Dont't forget to increase your input dimensionality by the size of your window.");

	float error = -1;
	unsigned long long seed = 0;
	int a;

	std::vector<int> input_indices = training_generator->getInputIndicesReference();
	std::vector<int> output_indices = training_generator->getOutputIndicesReference();
	Vector* target_values = training_generator->getTargetValuesReference();
	
	this->mlp->forward(input_indices,output_indices);

	pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
	
	for (a = 0; a < num_threads; a++) {
		
		StochasticGradientWorker* worker = new StochasticGradientWorker(this->mlp, this->criterion, training_generator, alpha, iterations, a, num_threads);
		pthread_create(&pt[a], NULL, TrainModelThread, worker);
		this->workers.push_back(worker);

	}

  	for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);

	for(a = 0; a < this->workers.size(); a++) {
		this->workers[a]->destroy(true);
	}

	this->workers.clear();
	
	return error;
}