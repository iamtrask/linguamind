/* File: linguamind.i */
%module linguamind

%{
#include "test.h"
#include <exception>
#include <vector>
%}

%include "test.h"
%include "std_except.i"
%include "std_vector.i"

namespace std {
   %template(vectori) vector<int>;
   %template(vectord) vector<double>;
};

%extend wrapped_array {
  inline size_t __len__() const { return N; }

  inline const Type& __getitem__(size_t i) const throw(std::out_of_range) {
    if (i >= N)
      throw std::out_of_range("out of bounds access");
    return self->data[i];
  }

  inline void __setitem__(size_t i, const Type& v) throw(std::out_of_range) {
    if (i >= N)
      throw std::out_of_range("out of bounds access");
    self->data[i] = v;
  }
}

%template (intArray40) wrapped_array<int, 40>;
%template (doubleArray15) wrapped_array<double, 15>;

%{
#define SWIG_FILE_WITH_INIT
#include "linguamind.h"
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
		std::vector<int> rollupSegmentNewVocab(char* key);
};

char* fact(char * text);