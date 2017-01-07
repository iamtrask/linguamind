#ifndef LAYER
#define LAYER

#include "../linalg/tensor.h"

class Layer {

	public:
		Layer();

		Tensor* weights;
		Tensor* output;
};


#endif