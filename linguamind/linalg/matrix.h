#ifndef MATRIX_DEF
#define MATRIX_DEF

#include <vector>
#include <iostream>

#include "vector.h"
#include "seed.h"

class Matrix {

	public:
		Matrix(int rows, int cols);
		
		int rows, cols;

		std::vector<Vector*> _data;

		Matrix zero();
		Matrix uniform(Seed* seed);

		// std::vector<float> get();
		// Vector* get(int i);
		// Vector* set(int i, Vector* x);

		Matrix operator*=(float x) const;
		Matrix operator/=(float x) const;
		Matrix operator+=(float x) const;
		Matrix operator-=(float x) const;

		Matrix operator*=(Matrix* x) const;
		Matrix operator/=(Matrix* x) const;
		Matrix operator+=(Matrix* x) const;
		Matrix operator-=(Matrix* x) const;
};

#endif

