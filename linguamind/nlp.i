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

