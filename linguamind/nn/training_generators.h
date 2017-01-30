#ifndef TRAINING_GENERATOR
#define TRAINING_GENERATOR

#include <vector>
#include "../linalg/vector.h"
#include "../linalg/matrix.h"
#include "../nlp/vocab.h"
#include "layer.h"
#include "criterion.h"
#include "sequential.h"

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


#endif