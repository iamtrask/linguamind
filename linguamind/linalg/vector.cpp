#include "vector.h"
#include <assert.h>
#include <stdexcept> 

typedef float real;  

Vector::Vector(int size) {

	this->size = size;
	this->_data = new float[this->size];
	// this->_data = (float*)malloc(this->size * sizeof(float));
	this->zero();

}

Vector::Vector(std::vector<bool> data) {
	this->size = data.size();
	this->_data = new float[this->size];
	for(int i=0; i<this->size; i++) this->_data[i] = (float)data[i];
}

int Vector::destroy() {
	delete this->_data;
	this->size = 0;
	return 0;
}

Vector Vector::resize(int size) {
	printf("Resizing Vector from %i to %i", this->size, size);
	if(size >= this->size) {
		this->size = size;
		this->_data = (float*)realloc(this->_data, this->size*sizeof(float));
	} else {
		this->size = size;
	}
}

Vector Vector::uniform(Seed* seed) {
	for(int i=0; i<this->size; i++) {
		seed->seed = seed->seed * (unsigned long long)25214903917 + 11;
		this->_data[i] = (float)(((seed->seed & 0xFFFF) / (float)65536));
	}
	return *this;
}

Vector Vector::zero() {
	for (int i=0; i<this->size; i++) this->_data[i] = 0;
	return *this;
}

std::vector<float> Vector::get() {
	std::vector<float> out;
	for (int i=0; i<this->size; i++) {
		out.push_back(this->_data[i]);
	}
	return out;
}

float Vector::get(int i) {
	
	if(i >= this->size) throw std::runtime_error("OutOfBounds: Vector isn't that big.");

	return this->_data[i];
}

void Vector::set(int i, float x) {
	
	if(i >= this->size) throw std::runtime_error("OutOfBounds: Vector isn't that big.");
	
	this->_data[i] = x;
	
	
	
}

float Vector::dot(Vector* x) {
	
	if(x->size != this->size) throw std::runtime_error("Vector:dot(Vector* x): Vectors not the same size");

	float out = 0;
	for (int i=0; i<this->size; i++) {
		out += x->_data[i] * this->_data[i];
	}
	return out;
}

void Vector::doti(int i, Vector* x, Vector* y) {
	
	if(x->size != y->size) throw std::runtime_error("Vector::dot(Vector* x): Vectors not the same size");
	if(i >= this->size) throw std::runtime_error("IndexOutOfBounds: Vector isn't that big.");
	
	this->_data[i] = x->dot(y);
		// this->_data[i] = cblas_sdot(this->size, x->_data, 1, y->_data, 1);	
}

void Vector::dotiadd(int i, Vector* x, Vector* y) {
	
	if(x->size != y->size) throw std::runtime_error("Vector::dot(Vector* x): Vectors not the same size");
	if(i >= this->size) throw std::runtime_error("IndexOutOfBounds: Vector isn't that big.");

	this->_data[i] += x->dot(y);
		// this->_data[i] = cblas_sdot(this->size, x->_data, 1, y->_data, 1);
}

Vector Vector::set(Vector* x) {
	if(x->size != this->size) throw std::runtime_error("Vector Vector::set(Vector* x): Vectors not the same size");
	for(int i=0; i<this->size; i++) {
		this->_data[i] = x->_data[i];
	}
	return *this;
}

Vector Vector::set(Vector* x, float a) {
	if(x->size != this->size) throw std::runtime_error("Vector Vector::set(Vector* x, float a): Vectors not the same size");
	for(int i=0; i<this->size; i++) {
		this->_data[i] = x->_data[i] * a;
	}
	return *this;
}

Vector Vector::addi(int i, float a) {
	if(i >= this->size) throw std::runtime_error("Vector Vector::addi(int i, float a): attempt to index non-existant value");
	// for(int i=0; i<this->size; i++) {
	this->_data[i] += a;
	// }
	return *this;
}

Vector Vector::addi(Vector* x, float a) {
	if(x->size != this->size) throw std::runtime_error("Vector Vector::addi(Vector* x, float a): Vectors not the same size");
	for(int i=0; i<this->size; i++) {
		this->_data[i] += x->_data[i] * a;
	}
	return *this;
}

Vector Vector::subi(Vector* x, float a) {
	if(x->size != this->size) throw std::runtime_error("Vector Vector::subi(Vector* x, float a): Vectors not the same size");
	for(int i=0; i<this->size; i++) {
		this->_data[i] -= x->_data[i] * a;
	}
	return *this;
}

Vector Vector::muli(float x) {
	for(int i=0; i<this->size; i++) {
		this->_data[i] *= x;
	}
	return *this;
}

Vector Vector::operator*=(float x) const {
	for(int i=0; i<this->size; i++) {
		this->_data[i] *= x;
	}
	return *this;
}

Vector Vector::divi(float x) {
	for(int i=0; i<this->size; i++) {
		this->_data[i] /= x;
	}
	return *this;
}

Vector Vector::operator/=(float x) const {
	for(int i=0; i<this->size; i++) {
		this->_data[i] /= x;
	}
	return *this;
}

Vector Vector::addi(float x) {

	for(int i=0; i<this->size; i++) {
		this->_data[i] += x;
	}
	return *this;
}

Vector Vector::operator+=(float x) const {
	for(int i=0; i<this->size; i++) {
		this->_data[i] += x;
	}
	return *this;
}

Vector Vector::subi(float x) {
	for(int i=0; i<this->size; i++) {
		this->_data[i] -= x;
	}
	return *this;
}

Vector Vector::operator-=(float x) const {
	for(int i=0; i<this->size; i++) {
		this->_data[i] -= x;
	}
	return *this;
}

Vector Vector::gei(float x) {
	for(int i=0; i<this->size; i++) {
		if(this->_data[i] >= x) {
			this->_data[i] = 1.0;
		} else {
			this->_data[i] = 0.0;
		}
	}
	return *this;
}

Vector Vector::operator>=(float x) const {
	for(int i=0; i<this->size; i++) {
		if(this->_data[i] >= x) {
			this->_data[i] = 1.0;
		} else {
			this->_data[i] = 0.0;
		}
	}
	return *this;
}

Vector Vector::lei(float x) {
	for(int i=0; i<this->size; i++) {
		if(this->_data[i] <= x) {
			this->_data[i] = 1.0;
		} else {
			this->_data[i] = 0.0;
		}
	}
	return *this;
}

Vector Vector::operator<=(float x) const {
	for(int i=0; i<this->size; i++) {
		if(this->_data[i] <= x) {
			this->_data[i] = 1.0;
		} else {
			this->_data[i] = 0.0;
		}
	}
	return *this;
}

Vector Vector::gti(float x) {
	for(int i=0; i<this->size; i++) {
		if(this->_data[i] > x) {
			this->_data[i] = 1.0;
		} else {
			this->_data[i] = 0.0;
		}
	}
	return *this;
}

Vector Vector::operator>(float x) const {
	for(int i=0; i<this->size; i++) {
		if(this->_data[i] > x) {
			this->_data[i] = 1.0;
		} else {
			this->_data[i] = 0.0;
		}
	}
	return *this;
}

Vector Vector::lti(float x) {
	for(int i=0; i<this->size; i++) {
		if(this->_data[i] < x) {
			this->_data[i] = 1.0;
		} else {
			this->_data[i] = 0.0;
		}
	}
	return *this;
}

Vector Vector::operator<(float x) const {
	for(int i=0; i<this->size; i++) {
		if(this->_data[i] < x) {
			this->_data[i] = 1.0;
		} else {
			this->_data[i] = 0.0;
		}
	}
	return *this;
}

Vector Vector::muli(Vector* x) {
	if(x->size != this->size) throw std::runtime_error("Vector Vector::muli(Vector* x): Vectors not the same size");
	for(int i=0; i<this->size; i++) {
		this->_data[i] *= x->_data[i];
	}
	return *this;
}

Vector Vector::operator*=(Vector* x) const {
	if(x->size != this->size) throw std::runtime_error("Vector Vector::operator*=(Vector* x): Vectors not the same size");
	for(int i=0; i<this->size; i++) {
		this->_data[i] *= x->_data[i];
	}
	return *this;
}

Vector Vector::divi(Vector* x) {
	if(x->size != this->size) throw std::runtime_error("Vector Vector::divi(Vector* x): Vectors not the same size");
	for(int i=0; i<this->size; i++) {
		this->_data[i] /= x->_data[i];
	}
	return *this;
}

Vector Vector::operator/=(Vector* x) const {
	if(x->size != this->size) throw std::runtime_error("Vector Vector::operator/=(Vector* x): Vectors not the same size");
	for(int i=0; i<this->size; i++) {
		this->_data[i] /= x->_data[i];
	}
	return *this;
}

Vector Vector::addi(Vector* x) {
	if(x->size != this->size) throw std::runtime_error("Vector Vector::addi(Vector* x): Vectors not the same size");
	for(int i=0; i<this->size; i++) {
		this->_data[i] += x->_data[i];
	}
	return *this;
}

Vector Vector::operator+=(Vector* x) const {
	if(x->size != this->size) throw std::runtime_error("Vector Vector::operator+=(Vector* x): Vectors not the same size");
	for(int i=0; i<this->size; i++) {
		this->_data[i] += x->_data[i];
	}
	return *this;
}

Vector Vector::subi(Vector* x) {
	if(x->size != this->size) throw std::runtime_error("Vector Vector::subi(Vector* x): Vectors not the same size");
	for(int i=0; i<this->size; i++) {
		this->_data[i] -= x->_data[i];
	}
	return *this;
}

Vector Vector::operator-=(Vector* x) const {
	if(x->size != this->size) throw std::runtime_error("Vector Vector::operator-=(Vector* x): Vectors not the same size");
	for(int i=0; i<this->size; i++) {
		this->_data[i] -= x->_data[i];
	}
	return *this;
}

Vector Vector::gei(Vector* x) {
	if(x->size != this->size) throw std::runtime_error("Vector Vector::gei(Vector* x): Vectors not the same size");
	for(int i=0; i<this->size; i++) {
		if(this->_data[i] >= x->_data[i]) {
			this->_data[i] = 1.0;
		} else {
			this->_data[i] = 0.0;
		}
	}
	return *this;
}

Vector Vector::operator>=(Vector* x) const {
	if(x->size != this->size) throw std::runtime_error("Vector Vector::operator>=(Vector* x): Vectors not the same size");
	for(int i=0; i<this->size; i++) {
		if(this->_data[i] >= x->_data[i]) {
			this->_data[i] = 1.0;
		} else {
			this->_data[i] = 0.0;
		}
	}
	return *this;
}

Vector Vector::lei(Vector* x) {
	if(x->size != this->size) throw std::runtime_error("Vector Vector::lei(Vector* x): Vectors not the same size");
	for(int i=0; i<this->size; i++) {
		if(this->_data[i] <= x->_data[i]) {
			this->_data[i] = 1.0;
		} else {
			this->_data[i] = 0.0;
		}
	}
	return *this;
}

Vector Vector::operator<=(Vector* x) const {
	if(x->size != this->size) throw std::runtime_error("Vector Vector::operator<=(Vector* x): Vectors not the same size");
	for(int i=0; i<this->size; i++) {
		if(this->_data[i] <= x->_data[i]) {
			this->_data[i] = 1.0;
		} else {
			this->_data[i] = 0.0;
		}
	}
	return *this;
}

Vector Vector::gti(Vector* x) {
	if(x->size != this->size) throw std::runtime_error("Vector Vector::gti(Vector* x): Vectors not the same size");
	for(int i=0; i<this->size; i++) {
		if(this->_data[i] > x->_data[i]) {
			this->_data[i] = 1.0;
		} else {
			this->_data[i] = 0.0;
		}
	}
	return *this;
}

Vector Vector::operator>(Vector* x) const {
	if(x->size != this->size) throw std::runtime_error("Vector Vector::operator>(Vector* x): Vectors not the same size");
	for(int i=0; i<this->size; i++) {
		if(this->_data[i] > x->_data[i]) {
			this->_data[i] = 1.0;
		} else {
			this->_data[i] = 0.0;
		}
	}
	return *this;
}

Vector Vector::lti(Vector* x) {
	if(x->size != this->size) throw std::runtime_error("Vector Vector::lti(Vector* x): Vectors not the same size");
	for(int i=0; i<this->size; i++) {
		if(this->_data[i] < x->_data[i]) {
			this->_data[i] = 1.0;
		} else {
			this->_data[i] = 0.0;
		}
	}
	return *this;
}

Vector Vector::operator<(Vector* x) const {
	if(x->size != this->size) throw std::runtime_error("Vector Vector::operator<(Vector* x): Vectors not the same size");
	for(int i=0; i<this->size; i++) {
		if(this->_data[i] < x->_data[i]) {
			this->_data[i] = 1.0;
		} else {
			this->_data[i] = 0.0;
		}
	}
	return *this;
}
