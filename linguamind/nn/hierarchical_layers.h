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

class LinearTree: public Layer  {
	

	private:
		bool sparse_output;
		bool sparse_input;

		int input_dim;
		int output_dim;

		Vector* output;
		Vector* input_grad;

		std::vector<int> output_indices;
		std::vector<int> full_output_indices;


		static bool comparePairs(const std::pair<float, int32_t>&,
                             const std::pair<float, int32_t>&);
	public:
		LinearTree(int input_dim, int output_dim);
		void init(int input_dim, int output_dim);

		std::vector<Node> tree;
		std::vector< std::vector<int32_t> > paths;
    	std::vector< std::vector<bool> > codes;
    	std::vector< std::vector<float> > factored_output;
    	std::vector<std::pair<float, int32_t>> heap;

		Matrix* weights;

		float sigmoid(float x);

		void createBinaryTree();

		int getPathSize(int i);
		int getCodeSize(int i);

		int getPath(int i, int j);
		bool getCode(int i, int j);

		Layer* duplicateWithSameWeights();

		int getInputDim();
		int getOutputDim();

		bool hasSparseInput();
		bool hasSparseOutput();

		std::vector<int> getOutputIndices();

		Vector* getOutput();
		Vector* getInputGrad();

		std::vector<int> getFullOutputIndices();

		void dfs(int32_t, int32_t, float,
             std::vector<std::pair<float, int32_t>>&,
             Vector*, int);

		int predict(Vector* input, std::vector<int> output_indices);
		int updateOutput(Vector* input, std::vector<int> &output_indices);
		int updateInputGrad(Vector* output_grad);
		int accGradParameters(Vector* input, Vector* output_grad, float alpha);
};

#endif