/* File: linalg.i */
%module linalg

%{
#include <vector>
%}

%include "std_vector.i"

namespace std {
   %template(vectori) vector<int>;
   %template(vectorf) vector<float>;
   %template(vectord) vector<double>;
};

%{
#define SWIG_FILE_WITH_INIT
#include "linalg/seed.h"
#include "linalg/vector.h"
#include "linalg/matrix.h"
#include "linalg/tensor.h"
%}

%include "cpointer.i"

/* Wrap a class interface around an "int *" */
%pointer_class(char, charp);


%exception {
    try {
        $action
    }
    catch (const std::runtime_error & e)
    {
        PyErr_SetString(PyExc_RuntimeError, const_cast<char*>(e.what()));
        return NULL;
    }
}


class Seed{

	public:
		Seed(unsigned long long seed);
		unsigned long long seed;
		void eat();
};

%extend Vector {
    float __getitem__(int i) {
    	if(i >= $self->size) {
			throw std::runtime_error("OutOfBounds: Index does not exist");
		}
         return $self->_data[i];
    }

    void __setitem__(int key, float item) {
    	if(key >= $self->size) {
			throw std::runtime_error("OutOfBounds: Index does not exist");
		}
        $self->_data[key] = item;
    }

    int __len__() {
        return $self->size;
    }

};

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


%extend Matrix {
    Vector* __getitem__(int i) {
    	if(i >= $self->rows) {
			throw std::runtime_error("OutOfBounds: Row does not exist");
		} 
         return $self->_data[i];
    }

    void __setitem__(int key, Vector* item) {
    	if(key >= $self->rows) {
			throw std::runtime_error("OutOfBounds: Row does not exist");
		}
        $self->_data[key] = item;
    }

    int __len__() {
        return $self->rows * $self->cols;
    }

};


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



