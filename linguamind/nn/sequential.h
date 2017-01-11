#ifndef SEQUENTIAL
#define SEQUENTIAL

#include "../linalg/vector.h"
#include "../linalg/matrix.h"
#include "sparse_linear.h"
#include "layer.h"
#include <vector>
#include <iostream>
#include <memory>

class Sequential  {

	public:
		Sequential(std::vector<Layer*> layers);

		std::vector<Layer*> layers;
		Vector* output;

		Layer* get(int i);
		Vector* forward(std::vector<int> &input_indices,std::vector<int> &output_indices);
		void backward(Vector* grad, std::vector<int> &output_indices);
};


#endif