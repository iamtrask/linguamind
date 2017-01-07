#ifndef TENSOR
#define TENSOR

#include <vector>
#include <iostream>

class Tensor{

	public:
		Tensor();
		Tensor(std::vector<int> shape);
		Tensor(int rows, int cols);

		std::vector<int> shape;
		int ndims;

		float* _data;
		long num_elements;
		unsigned long long seed;

		void init(std::vector<int> shape);
		
		float dotRow(Tensor* a, int index);

		Tensor addRowi(Tensor* a, int index);

		Tensor sub(Tensor* a, Tensor* b);

		Tensor uniform();
		Tensor zero();

		float get(int x);
		void set(int x, float y);

		Tensor operator*=(float x) const;
		Tensor operator/=(float x) const;
		Tensor operator+=(float x) const;
		Tensor operator-=(float x) const;
};

#endif