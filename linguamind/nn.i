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
#include "nn/lstm.h"
#include "nn/relu.h"
#include "nn/sigmoid.h"
#include "nn/tanh.h"
#include "nn/criterion.h"
#include "nn/sequential.h"
#include "nn/training_generators.h"
#include "nn/stochastic_gradient.h"
#include "nn/hierarchical_layers.h"
#include "nlp/vocab.h"
%}
#include "nn/layer.h"
#include "nlp/vocab.h"

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

		virtual Layer* duplicateWithSameWeights();

		virtual int getInputDim() = 0;
		virtual int getOutputDim() = 0;

		virtual bool hasSparseInput() = 0;
		virtual bool hasSparseOutput() = 0;

		virtual Vector* getOutput() = 0;
		virtual Vector* getInputGrad() = 0;

		virtual std::vector<int> getFullOutputIndices() = 0;

		virtual int updateOutput(Vector*, std::vector<int> &sparse_output) = 0;
		virtual int updateInputGrad(Vector* output_grad) = 0;
		virtual int accGradParameters(Vector* input, Vector* output_grad, float alpha) = 0;
};


class FlexLayer {

	private:

		int input_dim;
		int output_dim;

		bool input_must_be_sparse;
		bool output_must_be_sparse;

		bool mandatory_identical_input_output_sparsity;

		bool contains_layers;

		Vector* output;
		Vector* input_grad;
		Vector* output_grad;

		std::vector<int> input_indices;
		std::vector<int> output_indices;

	public:
		
		FlexLayer() {
    		
  		}
  		virtual ~FlexLayer() {

		}

		virtual int destroy(bool dont_destroy_weights) = 0;

		virtual FlexLayer* duplicateWithSameWeights() = 0;

		virtual int getInputDim() = 0;
		virtual int getOutputDim() = 0;

		virtual bool inputMustBeSparse() = 0;
		virtual bool outputMustBeSparse() = 0;
		virtual bool mandatoryIdenticalInputOutputSparsity() = 0;

		virtual bool containsLayers() = 0;

		virtual Vector* getOutput() = 0;
		virtual std::vector<int> &getOutputIndices() = 0;
		virtual Vector* getInputGrad() = 0;
		virtual int setOutputGrad(Vector* output_grad) = 0;

		virtual int updateOutputDenseToDense(Vector* input) = 0;
		virtual int updateOutputDenseToWeightedSparse(Vector* input, std::vector<int> &sparse_output) = 0;
		virtual int updateOutputWeightedSparseToDense(Vector* input, std::vector<int> &sparse_input) = 0;
		virtual int updateOutputWeightedSparseToWeightedSparse(Vector* input, std::vector<int> &sparse_input, std::vector<int> &sparse_output) = 0;
		virtual int updateOutputBinarySparseToDense(std::vector<int> &sparse_input) = 0;
		virtual int updateOutputBinarySparseToWeightedSparse(std::vector<int> &sparse_input, std::vector<int> &sparse_output) = 0;

		virtual int backward(Vector* output_grad) = 0;
		virtual int updateInputGrad(Vector* output_grad) = 0;
		virtual int accGradParameters(float alpha) = 0;

		virtual int reset() = 0;
};

// THIS IS SURPRISINGLY IMPORTANT TO MAKE THESE VALUES ACCEPTABLE AS PARAMETERS
// not as LayerBuilder explicitly persay... but to be able to pass in vector<Layer*> you need it.
namespace std {
   %template(LayerBuilder) vector<Layer*>;
   %template(FlexLayerBuilder) vector<FlexLayer*>;
   %template(TrainingExample) vector<vector<int> >;
};

class FlexLinear: public FlexLayer {

	private:

		int forward_code;

		bool weights_configured_input_sparse;

		int input_dim;
		int output_dim;

		bool output_must_be_sparse;
		bool input_must_be_sparse;

		bool mandatory_identical_input_output_sparsity;

		bool contains_layers;

		Vector* input;
		Vector* output;

		Vector* input_grad;
		Vector* output_grad;

		std::vector<int> input_indices;
		std::vector<int> output_indices;

	public:
		FlexLinear(int input_dim, int output_dim);
		FlexLinear(int input_dim, int output_dim, bool init_weights);
		// ~FlexLinear();
		void init(int input_dim, int output_dim, bool init_weights);

		int destroy(bool dont_destroy_weights);

		Matrix* weights;

		FlexLayer* duplicateWithSameWeights();

		int swapInputOutputSparsity();
		bool mandatoryIdenticalInputOutputSparsity();

		int getInputDim();
		int getOutputDim();

		bool inputMustBeSparse();
		bool outputMustBeSparse();

		bool containsLayers();

		Vector* getOutput();
		std::vector<int> &getOutputIndices();
		Vector* getInputGrad();
		int setOutputGrad(Vector* output_grad);

		// int updateOutput(Vector* input, std::vector<int> &not_used);
		int updateOutputDenseToDense(Vector* input);
		int updateOutputDenseToWeightedSparse(Vector* input, std::vector<int> &output_indices);
		int updateOutputWeightedSparseToDense(Vector* input, std::vector<int> &input_indices);
		int updateOutputWeightedSparseToWeightedSparse(Vector* input, std::vector<int> &input_indices, std::vector<int> &output_indices);
		int updateOutputBinarySparseToDense(std::vector<int> &input_indices);
		int updateOutputBinarySparseToWeightedSparse(std::vector<int> &input_indices, std::vector<int> &output_indices);

		int backward(Vector* output_grad);
		int updateInputGrad(Vector* output_grad);
		int accGradParameters(float alpha);

		int reset();
};

class FlexLSTMModule: public FlexLayer {

	private:

		int forward_code;

		bool weights_configured_input_sparse;

		int input_dim;
		int output_dim;

		bool output_must_be_sparse;
		bool input_must_be_sparse;

		bool mandatory_identical_input_output_sparsity;

		bool contains_layers;

		int batch_size;
		int batch_i;

		Vector* input;
		Vector* output;

		Vector* input_grad;
		Vector* output_grad;

		std::vector<int> input_indices;
		std::vector<int> output_indices;

	public:
		FlexLSTMModule(int input_dim, int output_dim, int batch_size);
		FlexLSTMModule(int input_dim, int output_dim, int batch_size, bool init_weights);
		// ~FlexLSTMModule();
		void init(int input_dim, int output_dim, int batch_size, bool init_weights);

		int destroy(bool dont_destroy_weights);

		Matrix* weights;

		// PARAMETERS

		Matrix* W_xg;
		Matrix* W_xi;
		Matrix* W_xf;
		Matrix* W_xo;

		Matrix* W_hi;
		Matrix* W_ho;

		Vector* b_g;
		Vector* b_i;
		Vector* b_o;

		// TEMP VARIABLES

		Vector* prev_h;
		Vector* prev_s;

		Vector* next_h_delta;
		Vector* next_s_delta;

		Vector* prev_h_delta;
		Vector* prev_s_delta;

		Vector* g;
		Vector* i;
		Vector* i_right_cache;
		Vector* f;
		Vector* o;
		Vector* s;
		Vector* h;

		Vector* g_delta;
		Vector* i_delta;
		Vector* f_delta;
		Vector* o_delta;
		Vector* s_delta;
		Vector* h_delta;

		Vector* gi_delta;

		float* sigmoidTable;

		FlexLayer* duplicateWithSameWeights();
		FlexLSTMModule* duplicateLSTMWithSameWeights();

		int swapInputOutputSparsity();
		bool mandatoryIdenticalInputOutputSparsity();

		int getInputDim();
		int getOutputDim();

		bool inputMustBeSparse();
		bool outputMustBeSparse();

		bool containsLayers();

		Vector* getOutput();
		std::vector<int> &getOutputIndices();
		Vector* getInputGrad();
		int setOutputGrad(Vector* output_grad);

		// int updateOutput(Vector* input, std::vector<int> &not_used);
		int updateOutputDenseToDense(Vector* input);
		int updateOutputDenseToWeightedSparse(Vector* input, std::vector<int> &output_indices);
		int updateOutputWeightedSparseToDense(Vector* input, std::vector<int> &input_indices);
		int updateOutputWeightedSparseToWeightedSparse(Vector* input, std::vector<int> &input_indices, std::vector<int> &output_indices);
		int updateOutputBinarySparseToDense(std::vector<int> &input_indices);
		int updateOutputBinarySparseToWeightedSparse(std::vector<int> &input_indices, std::vector<int> &output_indices);

		int backward(Vector* output_grad);
		int updateInputGrad(Vector* output_grad);
		int accGradParameters(float alpha);

		int reset();
};

class FlexLSTM: public FlexLayer {

	private:

		int forward_code;

		bool weights_configured_input_sparse;

		int input_dim;
		int output_dim;

		bool output_must_be_sparse;
		bool input_must_be_sparse;

		bool mandatory_identical_input_output_sparsity;

		bool contains_layers;

		Vector* input;
		Vector* output;

		Vector* input_grad;
		Vector* output_grad;

		Vector* default_prev_h;
		Vector* default_prev_s;

		std::vector<int> input_indices;
		std::vector<int> output_indices;

	public:
		FlexLSTM(int input_dim, int output_dim, int batch_size, int bptt);
		FlexLSTM(int input_dim, int output_dim, int batch_size, int bptt, bool init_weights);
		// ~FlexLSTM();
		void init(int input_dim, int output_dim, int batch_size, int bptt, bool init_weights);

		int destroy(bool dont_destroy_weights);
		int reset();

		int batch_size;

		int bptt;

		std::vector<FlexLSTMModule*> layers;

		int layer_i;

		int seq_i;

		Matrix* weights;

		FlexLayer* duplicateWithSameWeights();

		int swapInputOutputSparsity();
		bool mandatoryIdenticalInputOutputSparsity();

		int getInputDim();
		int getOutputDim();

		bool inputMustBeSparse();
		bool outputMustBeSparse();

		bool containsLayers();

		Vector* getOutput();
		std::vector<int> &getOutputIndices();
		Vector* getInputGrad();
		int setOutputGrad(Vector* output_grad);

		// int updateOutput(Vector* input, std::vector<int> &not_used);
		int updateOutputDenseToDense(Vector* input);
		int updateOutputDenseToWeightedSparse(Vector* input, std::vector<int> &output_indices);
		int updateOutputWeightedSparseToDense(Vector* input, std::vector<int> &input_indices);
		int updateOutputWeightedSparseToWeightedSparse(Vector* input, std::vector<int> &input_indices, std::vector<int> &output_indices);
		int updateOutputBinarySparseToDense(std::vector<int> &input_indices);
		int updateOutputBinarySparseToWeightedSparse(std::vector<int> &input_indices, std::vector<int> &output_indices);

		int backward(Vector* output_grad);
		int updateInputGrad(Vector* output_grad);
		int accGradParameters(float alpha);
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
		void init(int input_dim, int output_dim);

		Matrix* weights;

		Layer* duplicateWithSameWeights();

		int getInputDim();
		int getOutputDim();

		bool hasSparseInput();
		bool hasSparseOutput();

		Vector* getOutput();
		Vector* getInputGrad();

		std::vector<int> getFullOutputIndices();

		int updateOutput(Vector* input, std::vector<int> &input_indices);
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
		void init(int input_dim, int output_dim);

		Matrix* weights;

		Layer* duplicateWithSameWeights();

		int getInputDim();
		int getOutputDim();

		bool hasSparseInput();
		bool hasSparseOutput();

		Vector* getOutput();
		Vector* getInputGrad();

		std::vector<int> getFullOutputIndices();

		int updateOutput(Vector* input, std::vector<int> &output_indices);
		int updateInputGrad(Vector* output_grad);
		int accGradParameters(Vector* input, Vector* output_grad, float alpha);
};

class WeightedSparseLinearInput: public Layer {

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

		WeightedSparseLinearInput(int input_dim, int output_dim);

		void init(int input_dim, int output_dim);

		Matrix* weights;

		Layer* duplicateWithSameWeights();

		int getInputDim();
		int getOutputDim();

		bool hasSparseInput();
		bool hasSparseOutput();

		Vector* getOutput();
		Vector* getInputGrad();

		std::vector<int> getFullOutputIndices();
		
		int predict(Vector* input, std::vector<int> input_indices);
		int updateOutput(Vector* input, std::vector<int> &input_indices);
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
		void init(int input_dim, int output_dim);

		Matrix* weights;

		Layer* duplicateWithSameWeights();

		int getInputDim();
		int getOutputDim();

		bool hasSparseInput();
		bool hasSparseOutput();

		Vector* getOutput();
		Vector* getInputGrad();

		std::vector<int> getFullOutputIndices();

		int updateOutput(Vector* input, std::vector<int> &not_used);
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
		void init(int dim);

		Layer* duplicateWithSameWeights();

		int getInputDim();
		int getOutputDim();

		bool hasSparseInput();
		bool hasSparseOutput();

		Vector* getOutput();
		Vector* getInputGrad();

		std::vector<int> getFullOutputIndices();

		int updateOutput(Vector* input, std::vector<int> &output_indices);
		int updateInputGrad(Vector* output_grad);
		int accGradParameters(Vector* input, Vector* output_grad, float alpha);
};

class Sigmoid: public Layer {
	
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

		float* expTable;

	public:
		Sigmoid(int dim);
		void init(int dim);

		Layer* duplicateWithSameWeights();

		int getInputDim();
		int getOutputDim();

		bool hasSparseInput();
		bool hasSparseOutput();

		Vector* getOutput();
		Vector* getInputGrad();

		std::vector<int> getFullOutputIndices();

		int updateOutput(Vector* input, std::vector<int> &output_indices);
		int updateInputGrad(Vector* output_grad);
		int accGradParameters(Vector* input, Vector* output_grad, float alpha);
};

class FlexSigmoid: public FlexLayer {

	private:

		int input_dim;
		int output_dim;

		bool input_must_be_sparse;
		bool output_must_be_sparse;

		bool contains_layers;

		bool mandatory_identical_input_output_sparsity;

		int forward_code;

		Vector* output;
		Vector* input_grad;
		Vector* output_grad;

		std::vector<int> input_indices;
		std::vector<int> output_indices;

		float* expTable;

	public:
		
		FlexSigmoid(int dim);
		int destroy(bool dont_destroy_weights);
		
		FlexLayer* duplicateWithSameWeights();

		int getInputDim();
		int getOutputDim();

		bool inputMustBeSparse();
		bool outputMustBeSparse();

		bool containsLayers();

		bool mandatoryIdenticalInputOutputSparsity();

		Vector* getOutput();
		std::vector<int> &getOutputIndices();
		Vector* getInputGrad();
		int setOutputGrad(Vector* output_grad);

		int updateOutputDenseToDense(Vector* input);
		int updateOutputDenseToWeightedSparse(Vector* input, std::vector<int> &sparse_output);
		int updateOutputWeightedSparseToDense(Vector* input, std::vector<int> &sparse_input);
		int updateOutputWeightedSparseToWeightedSparse(Vector* input, std::vector<int> &sparse_input, std::vector<int> &sparse_output);
		int updateOutputBinarySparseToDense(std::vector<int> &sparse_input);
		int updateOutputBinarySparseToWeightedSparse(std::vector<int> &sparse_input, std::vector<int> &sparse_output);

		int backward(Vector* output_grad);
		int updateInputGrad(Vector* output_grad);
		int accGradParameters(float alpha);

		int reset();
};

class FlexTanh: public FlexLayer {

	private:

		int input_dim;
		int output_dim;

		bool input_must_be_sparse;
		bool output_must_be_sparse;

		bool contains_layers;

		bool mandatory_identical_input_output_sparsity;

		int forward_code;

		Vector* output;
		Vector* input_grad;
		Vector* output_grad;

		std::vector<int> input_indices;
		std::vector<int> output_indices;

		float* expTable;

	public:
		
		FlexTanh(int dim);
		int destroy(bool dont_destroy_weights);
		
		FlexLayer* duplicateWithSameWeights();

		int getInputDim();
		int getOutputDim();

		bool inputMustBeSparse();
		bool outputMustBeSparse();

		bool containsLayers();

		bool mandatoryIdenticalInputOutputSparsity();

		Vector* getOutput();
		std::vector<int> &getOutputIndices();
		Vector* getInputGrad();
		int setOutputGrad(Vector* output_grad);

		int updateOutputDenseToDense(Vector* input);
		int updateOutputDenseToWeightedSparse(Vector* input, std::vector<int> &sparse_output);
		int updateOutputWeightedSparseToDense(Vector* input, std::vector<int> &sparse_input);
		int updateOutputWeightedSparseToWeightedSparse(Vector* input, std::vector<int> &sparse_input, std::vector<int> &sparse_output);
		int updateOutputBinarySparseToDense(std::vector<int> &sparse_input);
		int updateOutputBinarySparseToWeightedSparse(std::vector<int> &sparse_input, std::vector<int> &sparse_output);

		int backward(Vector* output_grad);
		int updateInputGrad(Vector* output_grad);
		int accGradParameters(float alpha);

		int reset();
};

class MSECriterion {

	public:	
		MSECriterion();

		Vector* grad;

		MSECriterion* duplicate();
		void destroy();
		
		float forward(Vector* input, Vector* target);
		float forward(Vector* input, Vector* target, std::vector<int> &output_indices);

		Vector* backward(Vector* output, Vector* target);
		Vector* backward(Vector* output, Vector* target, std::vector<int> output_indices);
};


class Sequential  {

	public:
		Sequential(std::vector<Layer*> layers);

		std::vector<Layer*> layers;
		Vector* output;

		Sequential* duplicateWithSameWeights();

		Layer* get(int i);
		Vector* forward(std::vector<int> &input_indices,std::vector<int> &output_indices);
		void backward(Vector* grad, std::vector<int> &output_indices);
};


class FlexSequential: public FlexLayer  {

	private:
		
		bool output_must_be_sparse;
		bool input_must_be_sparse;

		bool contains_layers;

		bool mandatory_identical_input_output_sparsity;
		int layer_index_to_begin_using_sequence_output_indices;

		int forward_code;

		int input_dim;
		int output_dim;

		Vector* input;
		Vector* output;

		Vector* input_grad;
		Vector* output_grad;

		std::vector<int> input_indices;
		std::vector<int> output_indices;

		std::vector<int> not_used;

		
		int num_layers;

	public:

		std::vector<FlexLayer*> layers;

		FlexSequential(std::vector<FlexLayer*> layers);
		// ~FlexSequential();
		int destroy(bool dont_destroy_weights);

		void init(int input_dim, int output_dim);

		FlexLayer* duplicateWithSameWeights();
		FlexSequential* duplicateSequentialWithSameWeights();

		int getLayerIndexToBeginUsingSequenceOutputIndices();

		FlexLayer* get(int i);

		int getInputDim();
		int getOutputDim();

		bool inputMustBeSparse();
		bool outputMustBeSparse();

		bool containsLayers();
		bool mandatoryIdenticalInputOutputSparsity();

		Vector* getOutput();
		std::vector<int> &getOutputIndices();
		Vector* getInputGrad();
		int setOutputGrad(Vector* output_grad);

		Vector* forward(std::vector<int> input_indices,std::vector<int> output_indices);
		Vector* forward(Vector* input,std::vector<int> output_indices);
		Vector* forward(std::vector<int> input_indices);
		Vector* forward(Vector* input);
	
		// int updateOutput(Vector* input, std::vector<int> &not_used);
		int updateOutputDenseToDense(Vector* input);
		int updateOutputDenseToWeightedSparse(Vector* input, std::vector<int> &output_indices);
		int updateOutputWeightedSparseToDense(Vector* input, std::vector<int> &input_indices);
		int updateOutputWeightedSparseToWeightedSparse(Vector* input, std::vector<int> &input_indices, std::vector<int> &output_indices);
		int updateOutputBinarySparseToDense(std::vector<int> &input_indices);
		int updateOutputBinarySparseToWeightedSparse(std::vector<int> &input_indices, std::vector<int> &output_indices);

		int backward(Vector* output_grad);
		int updateInputGrad(Vector* output_grad);
		int accGradParameters(float alpha);

		int reset();

};

class TrainingGenerator {

	public:
		
		TrainingGenerator() {
    		
  		}
  		virtual ~TrainingGenerator() {

		}

		virtual TrainingGenerator* getCopyForSection(int starting, int ending) = 0;

		virtual int next() = 0;
		virtual bool hasNext() = 0;
		virtual int reset() = 0;
		virtual long getSize() = 0;
		virtual long getI() = 0;

		virtual bool shouldReset() = 0;
		virtual std::vector<int> &getInputIndicesReference() = 0;
		virtual std::vector<int> &getOutputIndicesReference() = 0;
		virtual Vector* getTargetValuesReference() = 0;

};

class SupervisedBinarySparseToWeightedSparse: public TrainingGenerator {

public:
	SupervisedBinarySparseToWeightedSparse(std::vector<std::vector<int> > input_indices, Vocab* input_vocab, std::vector<std::vector<int> > output_indices, Vocab* output_vocab);

	std::vector<std::vector<int> > input_indices;
	Vocab* input_vocab;

	std::vector<std::vector<int> > output_indices;
	Vocab* output_vocab;

	long size;

	long section_start;
	long section_end;

	long i;
	int pred_i;
	int window_len;

	bool has_next;

	TrainingGenerator* getCopyForSection(int starting, int ending);

	int next();
	bool hasNext();
	int reset();
	long getSize();
	long getI();

	std::vector<int> &getInputIndicesReference();
	std::vector<int> &getOutputIndicesReference();
	Vector* getTargetValuesReference();

	bool shouldReset();

};


class Sampler {
	public:
		Sampler(Vocab* vocab, int sample_size);

		Vocab* vocab;
		unsigned long long seed;

		int sample_size;
		int output_size;

		Vector* target_values;

		std::vector<int> next(std::vector<int> & output_indices);
		Vector* getTargetValues();

};


class CBOW: public TrainingGenerator {

	public:
		CBOW(std::vector<std::vector<int> > &window_indices,Vocab* vocab, Sampler* sampler, int window_left, int window_right, bool model_order);

		std::vector<std::vector<int> > window_indices;
		long window_indices_size;
		long size;

		std::vector<int> input_indices;
		std::vector<int> output_indices;

		Vocab* vocab;
		Sampler* sampler;
		
		int window_left;
		int window_right;

		bool model_order;
		int minimum_input_dimensionality;

		int iterator;
		int negative;

		int starting_win_i;
		int win_i;
		int pred_i;
		int window_len;

		unsigned long long seed;
		bool has_next;

		int next();
		bool hasNext();
		int reset();
		long getSize();
		long getI();

		TrainingGenerator* getCopyForSection(int starting, int ending);
		std::vector<int> &getInputIndicesReference();
		std::vector<int> &getOutputIndicesReference();
		Vector* getTargetValuesReference();
		bool shouldReset();

};


class StochasticGradientWorker{ 

	public:
		StochasticGradientWorker(FlexSequential* mlp, MSECriterion* criterion, TrainingGenerator* training_generator,float alpha, int iterations, int worker_id, int num_workers);

		FlexSequential* mlp;
		MSECriterion* criterion;
		TrainingGenerator* training_generator;

		float alpha;
		int iterations;

		int worker_id;
		int num_workers;

		void train();
		void destroy(bool dont_destroy_weights);

};

class StochasticGradient  {

	public:
		StochasticGradient(FlexSequential* mlp, MSECriterion* criterion);

		FlexSequential* mlp;
		MSECriterion* criterion;
		std::vector<StochasticGradientWorker*> workers;


		float train(TrainingGenerator* training_generator, float alpha, int iterations, int num_threads);
};

void *TrainModelThread(void *sgd);

class LinearTree: public FlexLayer  {
	

	private:

		int k;

		int input_dim;
		int output_dim;

		bool output_must_be_sparse;
		bool input_must_be_sparse;

		bool mandatory_identical_input_output_sparsity;

		bool contains_layers;

		Vector* input;
		Vector* output;

		Vector* input_grad;
		Vector* output_grad;

		std::vector<int> input_indices;
		std::vector<int> output_indices;


		static bool comparePairs(const std::pair<float, int32_t>&,
                             const std::pair<float, int32_t>&);
	public:
		LinearTree(int input_dim, int output_dim, int k);
		void init(int input_dim, int output_dim, int k);

		std::vector<Node> tree;
		std::vector< std::vector<int32_t> > paths;
    	std::vector< std::vector<bool> > codes;
    	std::vector< std::vector<float> > factored_output;
    	std::vector<std::pair<float, int32_t>> heap;

		Matrix* weights;

		// new
		int destroy(bool dont_destroy_weights);

		FlexLayer* duplicateWithSameWeights();

		int getInputDim();
		int getOutputDim();

		// new
		bool inputMustBeSparse();
		bool outputMustBeSparse();
		bool mandatoryIdenticalInputOutputSparsity();
		bool containsLayers();

		Vector* getOutput();

		// modified
		std::vector<int> &getOutputIndices();

		Vector* getInputGrad();
		
		// new
		int setOutputGrad(Vector* output_grad);

		float sigmoid(float x);

		void createBinaryTree();

		int getPathSize(int i);
		int getCodeSize(int i);

		int getPath(int i, int j);
		bool getCode(int i, int j);

		void dfs(int32_t, int32_t, float,
             std::vector<std::pair<float, int32_t>>&,
             Vector*, int);

		int predict(Vector* input, std::vector<int> output_indices);
		
		// new
		int updateOutputDenseToDense(Vector* input);
		int updateOutputDenseToWeightedSparse(Vector* input, std::vector<int> &sparse_output);
		int updateOutputWeightedSparseToDense(Vector* input, std::vector<int> &sparse_input);
		int updateOutputWeightedSparseToWeightedSparse(Vector* input, std::vector<int> &sparse_input, std::vector<int> &sparse_output);
		int updateOutputBinarySparseToDense(std::vector<int> &sparse_input);
		int updateOutputBinarySparseToWeightedSparse(std::vector<int> &sparse_input, std::vector<int> &sparse_output);

		// new
		int backward(Vector* output_grad);

		int updateInputGrad(Vector* output_grad);

		// modified
		int accGradParameters(float alpha);
};

