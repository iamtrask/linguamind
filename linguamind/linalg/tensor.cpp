#include "tensor.h"

Tensor::Tensor() {
	
}

Tensor::Tensor(std::vector<int> shape) {
	
	// stores the shape of the tensor
	this->shape = shape;

	// stores the number of dimensions in tensor
	this->ndims = this->shape.size();
	
	// calculates the size 
	this->num_elements = this->shape[0];
	for(int i=1; i<this->ndims; i++) this->num_elements *= shape[i];
	
	// initialize data values
	this->_data = (float*)malloc(this->num_elements * sizeof(float));

	this->seed = 0;
}

Tensor Tensor::addRowi(Tensor* a, int index) {
	for(int i=0; i < this->shape[1]; i++) {
		this->_data[i] += a->_data[i * this->shape[1] + i];
	}
	return *this;
}

float Tensor::dotRow(Tensor* a, int index) {
	float out = 0;
	for(int i=0; i < this->shape[1]; i++) {
		out += a->_data[i] * this->_data[index * this->shape[1] + i];
	}
	return out;
}

Tensor Tensor::uniform() {
	for(int i=0; i<this->num_elements; i++) {
		this->seed = this->seed * (unsigned long long)25214903917 + 11;
		this->_data[i] = (((this->seed & 0xFFFF) / (float)65536));
	}
	return *this;
}

Tensor Tensor::zero() {
	for(int i=0; i<this->num_elements; i++) {
		this->_data[i] = 0;
	}
	return *this;
}

float Tensor::get(int x) {
	return this->_data[x];
}

Tensor Tensor::operator*=(float x) const {
	for(int i=0; i<this->num_elements; i++) {
		this->_data[i] *= x;
	}
	return *this;
}

Tensor Tensor::operator/=(float x) const {
	for(int i=0; i<this->num_elements; i++) {
		this->_data[i] /= x;
	}
	return *this;
}

Tensor Tensor::operator+=(float x) const {
	for(int i=0; i<this->num_elements; i++) {
		this->_data[i] += x;
	}
	return *this;
}

Tensor Tensor::operator-=(float x) const {
	for(int i=0; i<this->num_elements; i++) {
		this->_data[i] -= x;
	}
	return *this;
}