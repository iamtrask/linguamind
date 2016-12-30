/* File: nlp.i */
%module nlp

%{
#include <vector>
%}

%include "std_vector.i"

namespace std {
   %template(vectori) vector<int>;
   %template(vectord) vector<double>;
};

%{
#define SWIG_FILE_WITH_INIT
#include "nlp/text.h"
%}

%include "cpointer.i"

/* Wrap a class interface around an "int *" */
%pointer_class(char, charp);

struct Term {
	char* letters;
};

class Tokenizer{
	public:
		Tokenizer();
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

	    std::unordered_map<std::string, std::vector<int> > sequences;
	    std::unordered_map<std::string, std::vector<char> > sequence_BIOs;
	    std::unordered_map<std::string, Vocab*> sequence_vocabs;

		Text(char* raw);
		Text(char* raw, char* bio);

		Vocab* createVocabForNewSequence(std::string key);
		void addSequence(std::string key, char* sequence, Vocab* vocab);
		std::vector<int> segmentSequenceIntoNewSequence(std::string raw_key, char* bio, std::string seg_key, Vocab* seg_vocab);

		Vocab* getVocab(char* key);
		std::vector<int> getSequence(char* key);

};