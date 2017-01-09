#include "stochastic_gradient.h"

StochasticGradient::StochasticGradient(Sequential* mlp, MSECriterion* criterion) {
	this->mlp = mlp;
	this->criterion = criterion;
}

float StochasticGradient::train(std::vector<int> input_indices, std::vector<int> output_indices, Vector* target_values, float alpha, int iterations) {
	for(int iter=0; iter<iterations; iter++) {
		Vector* pred = this->mlp->forward(input_indices,output_indices);
		// float error = this->criterion->forward(pred, target_values,output_indices);
		this->mlp->backward(this->criterion->backward(pred,target_values,output_indices),output_indices);

		Vector* prev_layer_output = NULL;
		for(int i=0; i<(int)this->mlp->layers.size()-1; i++) {
			this->mlp->get(i)->accGradParameters(prev_layer_output,this->mlp->get(i+1)->getInputGrad(),alpha);
			prev_layer_output = this->mlp->get(i)->getOutput();
		}

		this->mlp->get(this->mlp->layers.size()-1)->accGradParameters(this->mlp->get(this->mlp->layers.size()-2)->getOutput(),criterion->grad,alpha);
	}
	return 0;
}