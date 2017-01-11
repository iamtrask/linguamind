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
#include "nlp/vocab.h"
#include "nlp/text.h"
%}

%include "cpointer.i"

/* Wrap a class interface around an "int *" */
%pointer_class(char, charp);

%exception {
    try {
        $action
    }
    catch (const std::runtime_error & e)
    {
        PyErr_SetString(PyExc_RuntimeError, const_cast<char*>(e.what()));
        return NULL;
    }
}

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

