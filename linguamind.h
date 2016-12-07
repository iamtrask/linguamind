/* File: linguamind.h */

#include <iostream>

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

class Text {
	public:
		char* raw;
		unsigned long raw_size;
		char* getRaw();

		int num_segment_keys;
		unsigned long long * segment_keys;
		int * segment_buffer_sizes;
		int ** segments;

		bool addSegmentBIO(char* key, char* bio);
		Text(char* raw);
		unsigned long long hashSegmentKey(char* key);
		int getKeyIndexAddIfNew(char* key,int segment_length);
		int addSegmentAndKey(unsigned long long keyHash,int segment_length);
};

char* fact(char * text);