#include "stochastic_gradient.h"

StochasticGradient::StochasticGradient(Sequential* mlp, MSECriterion* criterion) {
	this->mlp = mlp;
	this->criterion = criterion;
}

float StochasticGradient::train(CBOW* training_generator, float alpha, int iterations) {
	
	float error;
	unsigned long long seed = 0;
	
	std::vector<int> input_indices = training_generator->getInputIndicesReference();
	std::vector<int> output_indices = training_generator->getOutputIndicesReference();
	Vector* target_values = training_generator->getTargetValuesReference();

	for(int iter=0; iter<iterations; iter++) {

		while(training_generator->has_next) {
			training_generator->next();
			Vector* pred = this->mlp->forward(input_indices,output_indices);
			error = this->criterion->forward(pred, target_values,output_indices);
			this->mlp->backward(this->criterion->backward(pred,target_values,output_indices),output_indices);

			Vector* prev_layer_output = NULL;	
			for(int i=0; i<(int)this->mlp->layers.size()-1; i++) {
				this->mlp->get(i)->accGradParameters(prev_layer_output,this->mlp->get(i+1)->getInputGrad(),alpha);
				prev_layer_output = this->mlp->get(i)->getOutput();
			}

			this->mlp->get(this->mlp->layers.size()-1)->accGradParameters(this->mlp->get(this->mlp->layers.size()-2)->getOutput(),criterion->grad,alpha);
		}

		training_generator->reset();
	}

	return error;
}