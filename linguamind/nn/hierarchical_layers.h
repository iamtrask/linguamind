#ifndef HIERARCHICAL_LAYERS
#define HIERARCHICAL_LAYERS

#include "../linalg/seed.h"
#include "../linalg/vector.h"
#include "../linalg/matrix.h"

#include "../nlp/vocab.h"

#include "layer.h"

#include <math.h>

#define MAX_CODE_LENGTH 40

struct Node {
  int32_t parent;
  int32_t left;
  int32_t right;
  int64_t count;
  bool binary;
  float output;
};

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

#endif