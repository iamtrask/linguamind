#ifndef TEXT
#define TEXT

#include <iostream>
#include <vector>

#include "vocab.h"

#define LONGESTKEY 13 // 9,223,372,036,854,775,807 - (26**13)
#define CHAR_VOCAB_SIZE 256

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

class Text {
	public:
		Text(char* filepath, Vocab* vocab);

		char* filepath;
		Vocab* vocab;

		std::vector<std::vector<int>> sentences;
		void ChangeVocab(Vocab* new_vocab);
		void cacheTokensInMemoryAsIndices();
		void ReadWord(char *word, FILE *fin);
};

#endif