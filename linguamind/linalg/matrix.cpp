#include "matrix.h"

#include <stdexcept> 

Matrix::Matrix(int rows, int cols) {

	this->rows = rows;
	this->cols = cols;

	for(int i=0; i<rows; i++) {
		this->_data.push_back(new Vector(cols));
	}

}

int Matrix::destroy() {
	for(int i=0; i<rows; i++) {
		this->_data[i]->destroy();	
	}
	this->_data.clear();
	this->rows = 0;
	this->cols = 0;
}

Matrix Matrix::zero() {
	for(int i=0; i < this->rows; i++) {
		this->_data[i]->zero();
	}
	return *this;
}


void Matrix::matmulset(Vector* input, Vector* output) {
	output->set(this->get(0),input->get(0));
	for(int index=1; index < input->size; index++) {
		output->addi(this->get(index),input->get(index));
	}
}

void Matrix::matmuladd(Vector* input, Vector* output) {

	for(int index=0; index < input->size; index++) {
		output->addi(this->get(index),input->get(index));
	}
}

void Matrix::Tmatmulset(Vector* input, Vector* output) {
	
	for(int index=0; index < input->size; index++) {
		input->doti(index, output, this->get(index));
	}

}

void Matrix::Tmatmuladd(Vector* input, Vector* output) {
	
	for(int index=0; index < input->size; index++) {
		input->dotiadd(index, output, this->get(index));
	}

}

Matrix Matrix::uniform(Seed* seed) {
	for(int i=0; i < this->rows; i++) {
		this->_data[i]->uniform(seed);
	}
	return *this;
}

Vector* Matrix::get(int i) {
	if(i >= this->rows) {
		throw std::runtime_error("Vector* Matrix::get(int i): OutOfBounds: Row does not exist.");
	} 
	return this->_data[i];
}

// change how the rows and columns are stored.
// this method is extremely inefficient... it's not supposed
// to be used during training or testing... only during setup.
void Matrix::transpose() {
	Matrix* t_mat = new Matrix(this->cols, this->rows);
	for(int i=0; i<this->rows; i++) {
		for(int j=0; j<this->cols; j++) {
			t_mat->get(j)->set(i,this->get(i)->get(j));
		}
	}

	this->_data.clear();

	this->_data = t_mat->_data;
	int rows = this->rows;
	int cols = this->cols;

	this->rows = cols;
	this->cols = rows;
}

void Matrix::muli(float x) {
	for(int i=0; i<this->rows; i++) {
		this->_data[i]->muli(x);
	}
}

void Matrix::subi(float x) {
	for(int i=0; i<this->rows; i++) {
		this->_data[i]->subi(x);
	}
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
		throw std::runtime_error("Matrix Matrix::operator*=(Matrix* x): OutOfBounds: Matrices not identically sized");
	} 
	for(int i=0; i<this->rows; i++) {
		this->_data[i]->muli(x->_data[i]);
	}
	return *this;
}

Matrix Matrix::operator/=(Matrix* x) const {
	if(x->rows != this->rows || x->cols != this->cols) {
		throw std::runtime_error("Matrix Matrix::operator/=(Matrix* x): OutOfBounds: Matrices not identically sized");
	} 
	for(int i=0; i<this->rows; i++) {
		this->_data[i]->divi(x->_data[i]);
	}
	return *this;
}

Matrix Matrix::operator+=(Matrix* x) const {
	if(x->rows != this->rows || x->cols != this->cols) {
		throw std::runtime_error("Matrix Matrix::operator+=(Matrix* x): OutOfBounds: Matrices not identically sized");
	} 
	for(int i=0; i<this->rows; i++) {
		this->_data[i]->addi(x->_data[i]);
	}
	return *this;
}

Matrix Matrix::operator-=(Matrix* x) const {
	if(x->rows != this->rows || x->cols != this->cols) {
		throw std::runtime_error("Matrix Matrix::operator-=(Matrix* x): OutOfBounds: Matrices not identically sized");
	} 
	for(int i=0; i<this->rows; i++) {
		this->_data[i]->subi(x->_data[i]);
	}
	return *this;
}