#ifndef SPARSE_LINEAR
#define SPARSE_LINEAR

#include <vector>
#include "../linalg/tensor.h"

class SparseLinear{

	public:
		SparseLinear(int, int);
		SparseLinear(int, int, bool);

		Tensor* weights;
		bool is_output_sparse;

		void init(int, int, bool);
};

#endif