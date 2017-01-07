#ifndef SEQUENTIAL
#define SEQUENTIAL

#include "layer.h"
#include "sparse_linear.h"
#include <vector>

class Sequential  {

	public:
		Sequential();

		std::vector<Layer*> layers;
		Tensor* output;

		void add(Layer* layer);
		Tensor* forward(std::vector<int> input_indices,std::vector<int> output_indices);
};


#endif