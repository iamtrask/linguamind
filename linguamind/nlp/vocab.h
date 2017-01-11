#ifndef TERM_VOCAB
#define TERM_VOCAB

#include <iostream>
#include <vector>

#define LONGESTKEY 13 // 9,223,372,036,854,775,807 - (26**13)
#define CHAR_VOCAB_SIZE 256


struct Term {
	char* letters;
};

class Vocab {
	public:
		Vocab();
		int size;

		int vocab_buffer_size;
		Term * vocab;

		int hash_size;
		int* hash_table;

		int addTerm(char* term);
		int getTermHash(char* term);
		Term getTermAtIndex(int i);
		int getTermIndex(char* term);
};

#endif