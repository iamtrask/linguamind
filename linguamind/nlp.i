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