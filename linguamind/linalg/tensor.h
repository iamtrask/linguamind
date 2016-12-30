#ifndef TENSOR
#define TENSOR

#include <vector>
#include <iostream>

class Tensor{

	public:
		Tensor();
		Tensor(std::vector<int> shape);
		std::vector<int> shape;
		int ndims;

		float* _data;
		long num_elements;
		unsigned long long seed;

		Tensor uniform();
		Tensor zero();

		float get(int x);

		Tensor operator*=(float x) const;
		Tensor operator/=(float x) const;
		Tensor operator+=(float x) const;
		Tensor operator-=(float x) const;
};

#endif