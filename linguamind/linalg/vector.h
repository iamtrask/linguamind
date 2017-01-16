#ifndef VECTOR
#define VECTOR

#include <vector>
#include <iostream>
#include "seed.h"

class Vector{

	public:
		Vector(int size);
		
		int size;

		float* _data;

		Vector resize(int size);

		Vector zero();
		Vector uniform(Seed* seed);

		std::vector<float> get();
		float get(int i);
		void set(int i, float x);

		float dot(Vector* x);
		void doti(int i, Vector* x, Vector* y);

		Vector set(Vector* x, float a);
		Vector addi(Vector* x, float a);
		Vector subi(Vector* x, float a);

		Vector muli(float x);
		Vector operator*=(float x) const;

		Vector divi(float x);
		Vector operator/=(float x) const;

		Vector addi(float x);
		Vector operator+=(float x) const;

		Vector subi(float x);
		Vector operator-=(float x) const;

		Vector gei(float x);
		Vector operator>=(float x) const;

		Vector lei(float x);
		Vector operator<=(float x) const;

		Vector gti(float x);
		Vector operator>(float x) const;

		Vector lti(float x);
		Vector operator<(float x) const;

		Vector muli(Vector* x);
		Vector operator*=(Vector* x) const;

		Vector divi(Vector* x);
		Vector operator/=(Vector* x) const;

		Vector addi(Vector* x);
		Vector operator+=(Vector* x) const;

		Vector subi(Vector* x);
		Vector operator-=(Vector* x) const;

		Vector gei(Vector* x);
		Vector operator>=(Vector* x) const;

		Vector lei(Vector* x);
		Vector operator<=(Vector* x) const;

		Vector gti(Vector* x);
		Vector operator>(Vector* x) const;

		Vector lti(Vector* x);
		Vector operator<(Vector* x) const;
};

#endif