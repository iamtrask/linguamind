#include "text.h"

Text::Text(char* filepath, Vocab* vocab) {
	
	this->filepath = filepath;
	this->vocab = vocab;

	// printf("\n\nFilepath: %s\n\n",filepath);

	this->cacheTokensInMemoryAsIndices();
}

// caches document assuming each newline is a new sentence
void Text::cacheTokensInMemoryAsIndices() {
	
	char word[MAX_STRING];
	int i,a;
	Term* t;
	FILE* fin = fopen(this->filepath, "rb");
	if (fin == NULL) {
    	throw std::runtime_error("ERROR: file not found");
    	exit(1);
	}
	std::vector<int> sentence;
	while (1) {
	    ReadWord(word, fin);
	    if (feof(fin)) break;

	    i = this->vocab->getTermIndex(word);
	    if (i == -1) {
	      i = this->vocab->addTerm(word);
	      this->vocab->vocab[i].count = 1;
	    } else{
	      this->vocab->vocab[i].count += 1;
	    } 

	    if(i == 0) {
	    	
	    	this->sentences.push_back(sentence);
	    	
	    	sentence.clear();
	    } else {
	    	sentence.push_back(i);
	    }
	    // if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
  	}
}

void Text::ChangeVocab(Vocab* new_vocab) {
	Vocab* old_vocab = this->vocab;
	int index;
	for(int sent_i=0; sent_i<this->sentences.size(); sent_i++) {
		for(int token_i=0; token_i < this->sentences[sent_i].size(); token_i++) {
			index = new_vocab->getTermIndex(old_vocab->vocab[this->sentences[sent_i][token_i]].letters);
			if(index == -1) {
				// this->sentences[sent_i].erase(this->sentences[sent_i].begin() + token_i);
				this->sentences[sent_i][token_i] = 0;
			} else {
				this->sentences[sent_i][token_i] = index;
			}
			
		}
	}

	this->vocab = new_vocab;
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void Text::ReadWord(char *word, FILE *fin) {
  int a = 0, ch;
  while (!feof(fin)) {
    ch = fgetc(fin);
    if (ch == 13) continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      if (ch == '\n') {
        strcpy(word, (char *)"</s>");
        return;
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  word[a] = 0;
}