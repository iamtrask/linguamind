#include "text.h"
#include <assert.h>

Tokenizer::Tokenizer() {
    // Matrix* m = new Matrix();
}

void Tokenizer::tokenize(Text* text) {
    // Matrix* m = new Matrix();
    // return m;
}

// this is the base "text" object, which can correlate to arbitrary text groupings
// this object is designed to be a sentence, paragraph, document, or corpus.
Text::Text(char* raw) {

    // default key is "text"
    Vocab* char_vocab = this->createVocabForNewSequence("text");
    this->addSequence("text",raw, char_vocab);
}

Text::Text(char* raw, char* bio) {

    // default key is "text"
    Vocab* char_vocab = this->createVocabForNewSequence("text");
    this->addSequence("text",raw, char_vocab);

    Vocab* token_vocab = this->createVocabForNewSequence("tokens");
    this->segmentSequenceIntoNewSequence("text",bio,"tokens",token_vocab);
}

std::vector<int> Text::segmentSequenceIntoNewSequence(std::string raw_key, char* bio, std::string seg_key, Vocab* seg_vocab) {
    
    std::vector<int> raw = this->sequences[raw_key];
    std::vector<int> sequence = this->sequences[seg_key];
    std::vector<int> start_end_indices;

    int start = -1;
    int end = -1;
    char * term; 
    int term_index;   

    for(int i=0; i<raw.size(); i++) {
        if(start == -1) {
            if(bio[i] == 'B') {
              start = i;  
            } 
        } else {
            if(bio[i] == 'B') {
                end = i;

                start_end_indices.push_back(start);
                start_end_indices.push_back(end);

                // start another
                start = i;
            } else if(bio[i] == 'I') {

            } else if(bio[i] == 'O') {
                end = i-1;

                start_end_indices.push_back(start);
                start_end_indices.push_back(end);

                start = -1;
            }
        }
    }

    if(start != -1) {
        end = raw.size()-1;
        start_end_indices.push_back(start);
        start_end_indices.push_back(end);       
    }

    int a = 0;
    int size = start_end_indices.size();
    while(a < size){
        start = start_end_indices[a];
        end = start_end_indices[a+1];

        // finish word
        term = (char*)malloc(end - start);
        for(int j=start; j <= end; j++) term[j-start] = raw[j];
        term_index = seg_vocab->addTerm(term);
        
        sequence.push_back(term_index);

        a++;
        a++;
    }

    this->sequences[seg_key] = sequence;
    
    return sequence;

}

Vocab* Text::getVocab(char * key) {
    std::string key_str(key);
    return this->sequence_vocabs[key_str];
}

std::vector<int> Text::getSequence(char* key) {
    std::string key_str(key);
    return this->sequences[key_str];
}

void Text::addSequence(std::string key, char* sequence, Vocab* vocab) {
    unsigned long size = strlen(sequence);
    for(unsigned long a=0; a < size; a++) this->sequences[key].push_back(sequence[a]);
}

Vocab* Text::createVocabForNewSequence(std::string key) {
    this->sequence_vocabs[key] = new Vocab();
    return this->sequence_vocabs[key];
}

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

