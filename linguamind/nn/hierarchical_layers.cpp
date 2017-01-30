#include "hierarchical_layers.h"

LinearTree::LinearTree(int input_dim, int output_dim, int k) {

	this->init(input_dim, output_dim, k);

}
void LinearTree::init(int input_dim, int output_dim, int k) {

	this->k = k;

	this->input_dim = input_dim; // embedding dim
	this->output_dim = output_dim; // output vocab

	this->input_must_be_sparse = false;
	this->output_must_be_sparse = true;

	this->mandatory_identical_input_output_sparsity = false;

	this->contains_layers = false;

	this->weights = new Matrix(output_dim, input_dim);
	this->weights->zero();

	this->input_grad = new Vector(this->input_dim);
	this->input_grad->zero();

	this->output = new Vector(this->output_dim);
	this->output->zero();

	// this->createBinaryTree();
}


int LinearTree::destroy(bool dont_destroy_weights) {
	// TODO
	return 0;
}

FlexLayer* LinearTree::duplicateWithSameWeights() {
	// TODO
	return NULL;
}



bool LinearTree::comparePairs(const std::pair<float, int32_t> &l,
                         const std::pair<float, int32_t> &r) {
  return l.first > r.first;
}

void LinearTree::dfs(int32_t k, int32_t node, float score,
                std::vector<std::pair<float, int32_t>>& heap,
                Vector* input, int depth) {
  if (heap.size() == k && score < heap.front().first) {
    return;
  }

  if (this->tree[node].left == -1 && this->tree[node].right == -1) {
    heap.push_back(std::make_pair(score, node));
    std::push_heap(heap.begin(), heap.end(), comparePairs);
    if (heap.size() > k) {
      std::pop_heap(heap.begin(), heap.end(), comparePairs);
      heap.pop_back();
    }
    return;
  }

  float f = this->sigmoid(input->dot(this->weights->get(node - this->output_dim)));
  tree[node].output = f;

  dfs(k, this->tree[node].left, score * (1.0 - f), heap, input, depth+1);
  dfs(k, this->tree[node].right, score * (f), heap, input, depth+1);

}

int LinearTree::predict(Vector* input, std::vector<int> output_indices) {
	
	// TODO
	this->updateOutputDenseToWeightedSparse(input,output_indices);
	return 0;
}

int LinearTree::updateOutputDenseToDense(Vector* input) {

	throw std::runtime_error("Error: Linear Tree must have sparse output.");

	return 0;
}
int LinearTree::updateOutputDenseToWeightedSparse(Vector* input, std::vector<int> &not_used) {\

	this->input = input;

	this->heap.clear();

	int k = this->k;
	dfs(k, 2 * this->output_dim - 2, 1.0, heap, input, 0);

	this->output_indices.clear();

	for(int i=0; i < k; i++) {
		this->output_indices.push_back(0);
	}

	for (int i=0; i<k; i++) {
		this->output_indices[i] = heap[i].second;
		this->output->set(i,heap[i].first);
	}

	return 0;
}
int LinearTree::updateOutputWeightedSparseToDense(Vector* input, std::vector<int> &sparse_input) {
	
	throw std::runtime_error("Error: Linear Tree must have sparse output.");

	return 0;
}
int LinearTree::updateOutputWeightedSparseToWeightedSparse(Vector* input, std::vector<int> &sparse_input, std::vector<int> &sparse_output) {

	throw std::runtime_error("TODO: Not yet implemented.");

	return 0;
}
int LinearTree::updateOutputBinarySparseToDense(std::vector<int> &sparse_input) {
	
	throw std::runtime_error("Error: Linear Tree must have sparse output.");

	return 0;
}
int LinearTree::updateOutputBinarySparseToWeightedSparse(std::vector<int> &sparse_input, std::vector<int> &sparse_output) {
	
	throw std::runtime_error("TODO: Not yet implemented.");

	return 0;
}

int LinearTree::backward(Vector* output_grad) {

	throw std::runtime_error("Error: backward called on non-compositional layer (doesn't have sub layers).");

	return 0;
}


// NOTE: this also perform the "word2vec shortcut", skipping the derivative of sigmoid
int LinearTree::updateInputGrad(Vector* output_grad) {
	

	int p;
	float c, loss,pred;
	int path_size;

	this->output_grad = output_grad;

	this->input_grad->zero(); // TODO: use Set for first iteration

	for(int i=0; i < this->output_indices.size(); i++) {
		
		std::vector<int32_t> path = this->paths[this->output_indices[i]];
		std::vector<bool> code = this->codes[this->output_indices[i]];

		path_size = path.size();
		
		loss = this->output_grad->get(this->output_indices[i]);

		for(int j=path_size-1; j >= 0; j--) {

			c = (float)code[j];
			p = (int)path[j];
			pred = this->tree[p + this->output_dim].output;

			if(c == 1) this->input_grad->addi(this->weights->get(p), -loss);
			if(c == 0) this->input_grad->addi(this->weights->get(p), loss);

			if(c == 1) {
				loss *= pred;
			} else {
				loss *= (1 - pred);
			}
			

		}
	}

	return 0;
}

// NOTE: this also perform the "word2vec shortcut", skipping the derivative of sigmoid
int LinearTree::accGradParameters(float alpha) {

	int p;
	float c, loss,pred;
	int path_size;


	for(int i=0; i < this->output_indices.size(); i++) {
		
		std::vector<int32_t> path = this->paths[output_indices[i]];
		std::vector<bool> code = this->codes[output_indices[i]];

		path_size = path.size();
		
		loss = this->output_grad->get(this->output_indices[i]);

		for(int j=path_size-1; j >= 0; j--) {

			c = (float)code[j];
			p = (int)path[j];
			pred = this->tree[p + this->output_dim].output;

			// reversed the sign of alpha so that it's actually a subtraction
			if(c == 1) this->weights->get(p)->addi(this->input,loss * alpha);
			if(c == 0) this->weights->get(p)->addi(this->input,-loss * alpha);

			if(c == 1) {
				loss *= pred;
			} else {
				loss *= (1 - pred);
			}
			

		}
	}

	return 0;
}

float LinearTree::sigmoid(float x) {
	return 1.0 / (1.0 + exp(-x)); // TODO: make this a lookup table... also link it with Sigmoid.cpp somehow
}

void LinearTree::createBinaryTree() {
	int redundancy = 1;
	int n_labels = this->output_dim;
	
	int osz_ = n_labels * redundancy;
  	int redundancy_ = redundancy;
  	int n_labels_ = n_labels;

	tree.resize(2 * osz_ - 1);
	for (int32_t i = 0; i < 2 * osz_ - 1; i++) {
	    tree[i].parent = -1;
	    tree[i].left = -1;
	    tree[i].right = -1;
	    tree[i].count = 1e15;
	    tree[i].binary = false;
	    tree[i].output = -1;
	  }

	  int32_t leaf = osz_ - 1;
	  int32_t node = osz_;
	  
	  for (int32_t i = osz_; i < 2 * osz_ - 1; i++) {
	    int32_t mini[2];
	    mini[0] = 0;
	    mini[1] = 0;
	    for (int32_t j = 0; j < 2; j++) {
	      // if (leaf >= 0 && tree[leaf].count < tree[node].count) {
        if(leaf >= 0) {
	        mini[j] = leaf--;
	      } else {
	        mini[j] = node++;
	      }
	    }
	    tree[i].left = mini[0];
	    tree[i].right = mini[1];
	    tree[i].count = tree[mini[0]].count + tree[mini[1]].count;
	    tree[mini[0]].parent = i;
	    tree[mini[1]].parent = i;
	    tree[mini[1]].binary = true;
	  }
	  for (int32_t i = 0; i < osz_; i++) {
	    std::vector<int32_t> path;
	    std::vector<bool> code;
	    int32_t j = i;
	    while (tree[j].parent != -1) {
	      path.push_back(tree[j].parent - osz_);
	      code.push_back(tree[j].binary);
	      j = tree[j].parent;
	    }
	    paths.push_back(path);
	    codes.push_back(code);
	  }
}

int LinearTree::getCodeSize(int i) {return this->codes[i].size();}
int LinearTree::getPathSize(int i) {return this->paths[i].size();}
bool LinearTree::getCode(int i, int j) {return this->codes[i][j];}
int LinearTree::getPath(int i, int j) {return this->paths[i][j];}
int LinearTree::getInputDim() { return this->input_dim;};
int LinearTree::getOutputDim() { return this->output_dim;};
bool LinearTree::inputMustBeSparse() {return this->input_must_be_sparse;};
bool LinearTree::outputMustBeSparse() {return this->output_must_be_sparse;}
bool LinearTree::mandatoryIdenticalInputOutputSparsity() {return this->mandatory_identical_input_output_sparsity;}
bool LinearTree::containsLayers(){ return this->contains_layers;}
Vector* LinearTree::getOutput() {return this->output;}
std::vector<int> &LinearTree::getOutputIndices() {return this->output_indices;}
Vector* LinearTree::getInputGrad() {return this->input_grad;}
int LinearTree::setOutputGrad(Vector* output_grad) {this->output_grad = output_grad; return 0;}