#include "stochastic_gradient.h"

StochasticGradientWorker::StochasticGradientWorker(Sequential* mlp, MSECriterion* criterion, CBOW* training_generator,float alpha, int iterations, int worker_id, int num_workers) {
	
	this->alpha = alpha;
	this->iterations = iterations;

	this->worker_id = worker_id;
	this->num_workers = num_workers;

	if(this->worker_id != 0) {
		this->mlp = mlp->duplicateWithSameWeights();
		this->criterion = criterion->duplicate();
	} else {
		this->mlp = mlp;
		this->criterion = criterion;
	}

	long batch_size_per_thread = training_generator->size / (long)num_workers;

	if(worker_id != num_workers - 1) {
		this->training_generator = training_generator->getCopyForSection(worker_id * batch_size_per_thread,(worker_id+1)*batch_size_per_thread);
	} else {
		this->training_generator = training_generator->getCopyForSection(worker_id * batch_size_per_thread,training_generator->size);
	}

}

void StochasticGradientWorker::train() {
	
	std::vector<int> input_indices = this->training_generator->getInputIndicesReference();
	std::vector<int> output_indices = this->training_generator->getOutputIndicesReference();
	Vector* target_values = this->training_generator->getTargetValuesReference();

	for(int iter=0; iter<this->iterations; iter++) {

		while(this->training_generator->has_next) {
			
			input_indices = this->training_generator->getInputIndicesReference();
			output_indices = this->training_generator->getOutputIndicesReference();
			target_values = this->training_generator->getTargetValuesReference();
			
			Vector* pred = this->mlp->forward(input_indices,output_indices);
			// error = this->criterion->forward(pred, target_values,output_indices);
			this->mlp->backward(this->criterion->backward(pred,target_values,output_indices),output_indices);

			Vector* prev_layer_output = NULL;	
			for(int i=0; i<(int)this->mlp->layers.size()-1; i++) {
				this->mlp->get(i)->accGradParameters(prev_layer_output,this->mlp->get(i+1)->getInputGrad(),this->alpha);
				prev_layer_output = this->mlp->get(i)->getOutput();
			}

			this->mlp->get(this->mlp->layers.size()-1)->accGradParameters(this->mlp->get(this->mlp->layers.size()-2)->getOutput(),criterion->grad,this->alpha);
			this->training_generator->next();

		}

		this->training_generator->reset();
	}
}

StochasticGradient::StochasticGradient(Sequential* mlp, MSECriterion* criterion) {
	this->mlp = mlp;
	this->criterion = criterion;
}

void *TrainModelThread(void *sgd_ptr) {

	StochasticGradientWorker* sgd_worker = ((StochasticGradientWorker *)sgd_ptr);
	sgd_worker->train();

	pthread_exit(NULL);
}

float StochasticGradient::train(CBOW* training_generator, float alpha, int iterations, int num_threads) {

	if(training_generator->minimum_input_dimensionality > this->mlp->get(0)->getInputDim()) throw std::runtime_error("ERROR: input layer too small for training example generator. Are you modeling order? Dont't forget to increase your input dimensionality by the size of your window.");

	float error = -1;
	unsigned long long seed = 0;
	int a;

	std::vector<int> input_indices = training_generator->getInputIndicesReference();
	std::vector<int> output_indices = training_generator->getOutputIndicesReference();
	Vector* target_values = training_generator->getTargetValuesReference();
	
	pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
	
	for (a = 0; a < num_threads; a++) {
		
		StochasticGradientWorker* worker = new StochasticGradientWorker(this->mlp, this->criterion, training_generator, alpha, iterations, a, num_threads);
		pthread_create(&pt[a], NULL, TrainModelThread, worker);

	}
  	for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);

	return error;
}