#include "vocab.h"
#include <assert.h>

Vocab::Vocab() {
    
    this->size = 0;

    this->vocab_buffer_size = 1000;
    this->vocab = (Term *) malloc(this->vocab_buffer_size * sizeof(Term));

    this->hash_size = 1000000;
    this->hash_table = (int *) malloc(this->hash_size * sizeof(int));
    for(int i=0; i<this->hash_size; i++) hash_table[i] = -1;

}

int Vocab::getTermIndex(char* term) {

    int hash = this->getTermHash(term);
    bool foundWord = false;

    while(this->hash_table[hash] != -1 && hash < this->hash_size) {
        
        // if two strings have the same length
        if(strlen(term) != strlen(this->vocab[this->hash_table[hash]].letters)) {
            hash += 1;
            continue;
        }

        // if the two strings have the same characters
        foundWord = true;
        for(int i=0; i<strlen(term); i++) {
            if(term[i] != this->vocab[this->hash_table[hash]].letters[i]) {
                foundWord = false;
                break;
            } 
        }

        // passed all the tests... return word index
        if(foundWord) {
            return this->hash_table[hash];
        }
    }

    // term doesn't exist
    return -1;
}

// adds the term to the vocabulary if it doesn't exist
// returns the index of the term (whether it's new or not)
int Vocab::addTerm(char* term) {
    int index = this->getTermIndex(term);
    if(index == -1) {
        this->size += 1;

        this->vocab[this->size - 1].letters = (char*)malloc(strlen(term)+1);
        strcpy(this->vocab[this->size - 1].letters,term);

        if(this->size > this->vocab_buffer_size - 10) {
            this->vocab_buffer_size *= 2;
            this->vocab = (Term*)realloc(this->vocab, this->vocab_buffer_size * sizeof(Term));
        }

        int hash = this->getTermHash(term);
        while(this->hash_table[hash] != -1) hash += 1;
        this->hash_table[hash] = this->size-1;

        return this->size-1;
    } else {
        return index;
    }
}

Term Vocab::getTermAtIndex(int i) {
    return this->vocab[i];
}

int Vocab::getTermHash(char* term) {
    unsigned long long hash = 0;
    for(int i=0; i<strlen(term); i++) {
        hash = ((hash + term[i]) * CHAR_VOCAB_SIZE) % this->hash_size;
    }
    return (int) hash;
}

