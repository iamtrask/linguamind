/* File: nn.i */
%module(directors="1") nn

%{
#include <vector>
#include <memory>
%}

%include "std_vector.i"

namespace std {
   %template(vectori) vector<int>;
   %template(vectord) vector<double>;
};

%{
#define SWIG_FILE_WITH_INIT
#include "linalg/seed.h"
#include "linalg/vector.h"
#include "linalg/matrix.h"
#include "nn/layer.h"
#include "nn/sparse_linear.h"
#include "nn/linear.h"
#include "nn/relu.h"
#include "nn/criterion.h"
#include "nn/sequential.h"
#include "nn/stochastic_gradient.h"
%}
#include "nn/layer.h"

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

// This cleans up the char ** array we malloc'd before the function call
%typemap(freearg) char ** {
  free((char *) $1);
}

class Layer {

	private:

		bool sparse_output;
		bool sparse_input;

		int input_dim;
		int output_dim;

		Vector* output;
		Vector* input_grad;

		std::vector<int> full_output_indices;

	public:
		
		Layer() {
    		
  		}
  		virtual ~Layer() {

		}

		virtual int getInputDim() = 0;
		virtual int getOutputDim() = 0;

		virtual bool hasSparseInput() = 0;
		virtual bool hasSparseOutput() = 0;

		virtual Vector* getOutput() = 0;
		virtual Vector* getInputGrad() = 0;

		virtual std::vector<int> getFullOutputIndices() = 0;

		virtual int updateOutput(Vector*, std::vector<int>) = 0;
		virtual int updateInputGrad(Vector* output_grad) = 0;
		virtual int accGradParameters(Vector* input, Vector* output_grad, float alpha) = 0;
};
namespace std {
   %template(LayerBuilder) vector<Layer*>;
};


class SparseLinearInput: public Layer {

	private:

		bool sparse_output;
		bool sparse_input;

		int input_dim;
		int output_dim;

		Vector* output;
		Vector* input_grad;

		std::vector<int> input_indices;
		std::vector<int> full_output_indices;

	public:
		SparseLinearInput(int input_dim, int output_dim);

		Matrix* weights;

		int getInputDim();
		int getOutputDim();

		bool hasSparseInput();
		bool hasSparseOutput();

		Vector* getOutput();
		Vector* getInputGrad();

		std::vector<int> getFullOutputIndices();

		int updateOutput(Vector* input, std::vector<int> input_indices);
		int updateInputGrad(Vector* output_grad);
		int accGradParameters(Vector* input, Vector* output_grad, float alpha);
};

class SparseLinearOutput: public Layer {

	private:
		bool sparse_output;
		bool sparse_input;

		int input_dim;
		int output_dim;

		Vector* output;
		Vector* input_grad;

		std::vector<int> output_indices;
		std::vector<int> full_output_indices;

	public:

		SparseLinearOutput(int input_dim, int output_dim);

		Matrix* weights;

		int getInputDim();
		int getOutputDim();

		bool hasSparseInput();
		bool hasSparseOutput();

		Vector* getOutput();
		Vector* getInputGrad();

		std::vector<int> getFullOutputIndices();

		int updateOutput(Vector* input, std::vector<int> output_indices);
		int updateInputGrad(Vector* output_grad);
		int accGradParameters(Vector* input, Vector* output_grad, float alpha);
};

class Linear: public Layer {

	private:
		
		bool sparse_output;
		bool sparse_input;

		int input_dim;
		int output_dim;

		Vector* output;
		Vector* input_grad;

		std::vector<int> full_output_indices;

	public:
		Linear(int input_dim, int output_dim);

		Matrix* weights;

		int getInputDim();
		int getOutputDim();

		bool hasSparseInput();
		bool hasSparseOutput();

		Vector* getOutput();
		Vector* getInputGrad();

		std::vector<int> getFullOutputIndices();

		int updateOutput(Vector* input, std::vector<int> not_used);
		int updateInputGrad(Vector* output_grad);
		int accGradParameters(Vector* input, Vector* output_grad, float alpha);
};

class Relu: public Layer {
	
	private:
		
		bool sparse_output;
		bool sparse_input;

		int input_dim;
		int output_dim;

		Vector* output;
		Vector* input_grad;

		Matrix* weights;

		std::vector<int> output_indices;
		std::vector<int> full_output_indices;

	public:
		Relu(int dim);

		int getInputDim();
		int getOutputDim();

		bool hasSparseInput();
		bool hasSparseOutput();

		Vector* getOutput();
		Vector* getInputGrad();

		std::vector<int> getFullOutputIndices();

		int updateOutput(Vector* input, std::vector<int> output_indices);
		int updateInputGrad(Vector* output_grad);
		int accGradParameters(Vector* input, Vector* output_grad, float alpha);
};

class MSECriterion {

	public:	
		MSECriterion();

		Vector* grad;

		float forward(Vector* input, Vector* target, std::vector<int> output_indices);
		Vector* backward(Vector* output, Vector* target, std::vector<int> output_indices);
};

class Sequential  {

	public:
		Sequential(std::vector<Layer*> layers);

		std::vector<Layer*> layers;
		Vector* output;

		Layer* get(int i);
		Vector* forward(std::vector<int> input_indices,std::vector<int> output_indices);
		void backward(Vector* grad, std::vector<int> output_indices);
};

class StochasticGradient  {

	public:
		StochasticGradient(Sequential* mlp, MSECriterion* criterion);

		Sequential* mlp;
		MSECriterion* criterion;

		float train(std::vector<int> input_indices, std::vector<int> output_indices, Vector* target_values, float alpha, int iterations);		
};
