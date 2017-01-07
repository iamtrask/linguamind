#include "matrix.h"

#include <stdexcept> 

Matrix::Matrix(int rows, int cols) {

	this->rows = rows;
	this->cols = cols;

	for(int i=0; i<rows; i++) {
		this->_data.push_back(new Vector(cols));
	}

}

Matrix Matrix::zero() {
	for(int i=0; i < this->rows; i++) {
		this->_data[i]->zero();
	}
	return *this;
}

Matrix Matrix::uniform(Seed* seed) {
	for(int i=0; i < this->rows; i++) {
		this->_data[i]->uniform(seed);
	}
	return *this;
}

Matrix Matrix::operator*=(float x) const {
	for(int i=0; i<this->rows; i++) {
		this->_data[i]->muli(x);
	}
	return *this;
}

Matrix Matrix::operator/=(float x) const {
	for(int i=0; i<this->rows; i++) {
		this->_data[i]->divi(x);
	}
	return *this;
}

Matrix Matrix::operator+=(float x) const {
	for(int i=0; i<this->rows; i++) {
		this->_data[i]->addi(x);
	}
	return *this;
}

Matrix Matrix::operator-=(float x) const {
	for(int i=0; i<this->rows; i++) {
		this->_data[i]->subi(x);
	}
	return *this;
}

Matrix Matrix::operator*=(Matrix* x) const {
	if(x->rows != this->rows || x->cols != this->cols) {
		throw std::runtime_error("OutOfBounds: Matrices not identically sized");
	} 
	for(int i=0; i<this->rows; i++) {
		this->_data[i]->muli(x->_data[i]);
	}
	return *this;
}

Matrix Matrix::operator/=(Matrix* x) const {
	if(x->rows != this->rows || x->cols != this->cols) {
		throw std::runtime_error("OutOfBounds: Matrices not identically sized");
	} 
	for(int i=0; i<this->rows; i++) {
		this->_data[i]->divi(x->_data[i]);
	}
	return *this;
}

Matrix Matrix::operator+=(Matrix* x) const {
	if(x->rows != this->rows || x->cols != this->cols) {
		throw std::runtime_error("OutOfBounds: Matrices not identically sized");
	} 
	for(int i=0; i<this->rows; i++) {
		this->_data[i]->addi(x->_data[i]);
	}
	return *this;
}

Matrix Matrix::operator-=(Matrix* x) const {
	if(x->rows != this->rows || x->cols != this->cols) {
		throw std::runtime_error("OutOfBounds: Matrices not identically sized");
	} 
	for(int i=0; i<this->rows; i++) {
		this->_data[i]->subi(x->_data[i]);
	}
	return *this;
}