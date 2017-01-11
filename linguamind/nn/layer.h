#ifndef LAYER
#define LAYER

#include "../linalg/seed.h"
#include "../linalg/vector.h"
#include "../linalg/matrix.h"

#include "layer.h"

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

		virtual int updateOutput(Vector*, std::vector<int> &sparse_output) = 0;
		virtual int updateInputGrad(Vector* output_grad) = 0;
		virtual int accGradParameters(Vector* input, Vector* output_grad, float alpha) = 0;
};
#endif