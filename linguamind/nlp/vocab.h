#ifndef TERM_VOCAB
#define TERM_VOCAB

#include <iostream>
#include <vector>

#define LONGESTKEY 13 // 9,223,372,036,854,775,807 - (26**13)
#define CHAR_VOCAB_SIZE 256


struct Term {
	char* letters;
	int cn;
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

		void sort(int min_count);
		int addTerm(char* term);
		unsigned int getTermHash(char* term);
		Term* getTermAtIndex(int i);
		int getTermIndex(char* term);

		int getUnigramValue(int index);
		void InitUnigramTable();
};

#endif