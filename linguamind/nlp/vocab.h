#ifndef TERM_VOCAB
#define TERM_VOCAB

#include <iostream>
#include <vector>

#include "../linalg/vector.h"

#define LONGESTKEY 13 // 9,223,372,036,854,775,807 - (26**13)
#define CHAR_VOCAB_SIZE 256


// struct Term {
// 	char* letters;
// 	int cn;
// };

struct Term {
  char *letters;
  int32_t parent;
  int32_t left;
  int32_t right;
  int64_t count;
  bool binary;
  float output;
};

class Vocab {
	public:
		Vocab();
		Vocab(Vocab* v);
		int size;

		int vocab_buffer_size;
		Term * vocab;

		int hash_size;
		int* hash_table;

		const int unigram_table_size = 1e8;
		int *unigram_table;


		std::vector<Term> tree;
		std::vector< std::vector<int32_t> > paths;
    	std::vector< Vector*> codes;
    	// std::vector< std::vector<float> > factored_output;
    	// std::vector<std::pair<float, int32_t>> heap;

		void sort(int min_count);
		int addTerm(char* term);
		unsigned int getTermHash(char* term);
		Term* getTermAtIndex(int i);
		int getTermIndex(char* term);

		int getUnigramValue(int index);
		void InitUnigramTable();
		void createBinaryTree();

		int getPathSize(int i);
		int getCodeSize(int i);

		int getPath(int i, int j);
		bool getCode(int i, int j);

		std::vector<int> &getPathReference(int i);
		Vector* getCodeReference(int i);
};

#endif