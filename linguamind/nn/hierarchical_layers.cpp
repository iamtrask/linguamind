#include "hierarchical_layers.h"

LinearTree::LinearTree(int input_dim, int output_dim) {

	this->init(input_dim, output_dim);

}
void LinearTree::init(int input_dim, int output_dim) {

	this->sparse_output = true;
	this->sparse_input = false;

	this->input_dim = input_dim; // embedding dim
	this->output_dim = output_dim; // output vocab

	this->weights = new Matrix(output_dim, input_dim);
	this->weights->zero();

	this->input_grad = new Vector(this->input_dim);
	this->input_grad->zero();

	this->output = new Vector(this->output_dim);
	this->output->zero();

	for(int i=0; i<this->output_dim; i++) this->full_output_indices.push_back(i);

	// this->createBinaryTree();
}

Layer* LinearTree::duplicateWithSameWeights() {

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

  // float f = (wo_->dotRow(hidden, node - osz_));
  float f = this->sigmoid(input->dot(this->weights->get(node - this->output_dim)));
  tree[node].output = f;
  // unsigned int temp_rand = rand_ * 902483 + 234;

  // if(f > 1.0) f = 1.0;
  // if(f < -1.0) f = -1.0;
  // f = (f + 1) / 2;

  // if(f > 0.5) loss_[depth] += f;
  // if(f <= 0.5) loss_[depth] += (1 - f);
  // nexamples_[depth] += 1;

  dfs(k, this->tree[node].left, score * (1.0 - f), heap, input, depth+1);
  dfs(k, this->tree[node].right, score * (f), heap, input, depth+1);

  // if(f < 0.5) {
  //   if(test) {
  //     if(f <= 0.55) dfs(k, semantic_tree_->tree[node].left, score * (1.0 - f), heap, hidden, test, depth+1);
  //     if (f >= 0.45) dfs(k, semantic_tree_->tree[node].right, score * (f), heap, hidden, test, depth+1);
  //   } else {
  //     if(f <= 0.55) dfs(k, semantic_tree_->tree[node].left, score * (1.0 - f), heap, hidden, test, depth+1);
  //     if (f >= 0.45) dfs(k, semantic_tree_->tree[node].right, score * (f), heap, hidden, test, depth+1);
  //   }
  // } else {
  //   if(test) {
  //     if (f >= 0.45) dfs(k, semantic_tree_->tree[node].right, score * (f), heap, hidden, test, depth+1);
  //     if(f <= 0.55) dfs(k, semantic_tree_->tree[node].left, score * (1.0 - f), heap, hidden, test, depth+1);
  //   } else {
  //     if (f >= 0.45) dfs(k, semantic_tree_->tree[node].right, score * (f), heap, hidden, test, depth+1);
  //     if(f <= 0.55) dfs(k, semantic_tree_->tree[node].left, score * (1.0 - f), heap, hidden, test, depth+1);
  //   }
  // }
  // if(temp_rand % 1000 > f * 1000) dfs(k, semantic_tree_->tree[node].left, score * (1.0 - f), heap, hidden);
  
  // temp_rand = temp_rand * 903917 + 11;
  // dfs(k, tree[node].left, score * (1.0 - f), heap, hidden);

  // if f is greater than 0, run this
  // if(temp_rand % 1000 > f * 1000) dfs(k, semantic_tree_->tree[node].right, score * (f), heap, hidden);
  
  // temp_rand = temp_rand * 903917 + 11;
  // dfs(k, tree[node].right, score * (f), heap, hidden);
}

int LinearTree::predict(Vector* input, std::vector<int> output_indices) {
	this->updateOutput(input,output_indices);
}

int LinearTree::updateOutput(Vector* input, std::vector<int> &output_indices) {

	this->heap.clear();

	int k = 2;
	dfs(k, 2 * this->output_dim - 2, 1.0, heap, input, 0);

	this->output_indices.clear();

	for (int i=0; i<k; i++) {
		this->output_indices.push_back(heap[i].second);
		this->output->set(heap[i].second,heap[i].first);
	}

	return 0;
}

int LinearTree::updateInputGrad(Vector* output_grad) {
	


	return 0;
}

int LinearTree::accGradParameters(Vector* input, Vector* output_grad, float alpha) {

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
bool LinearTree::hasSparseInput() {return this->sparse_input;};
bool LinearTree::hasSparseOutput() {return this->sparse_output;}
Vector* LinearTree::getOutput() {return this->output;}
std::vector<int> LinearTree::getOutputIndices() {return this->output_indices;}
Vector* LinearTree::getInputGrad() {return this->input_grad;}
std::vector<int> LinearTree::getFullOutputIndices() {return this->full_output_indices;}