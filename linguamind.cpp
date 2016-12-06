#include "linguamind.h"
#include <assert.h>

// this is the base "text" object, which can correlate to arbitrary text groupings
// this object is designed to be a sentence, paragraph, document, or corpus.
Text::Text(char* raw) {
    
    // this is the raw character data for the text
    this->raw_size = (unsigned long)strlen(raw)+1;
    this->raw = (char*)malloc(this->raw_size); 
    strcpy(this->raw, raw);

    // segments are BIO markers for regions of text
    // since you have multiple types of segments 
    // (i.e. token segment, sentence segment, document segment)
    // they are instead key->value pairs.
    // Each key is the name of a segment (i.e., "tokenseg")
    // and is correlated with one row of this->segments
    this->segments = (int **)malloc(0);
    this->segment_buffer_sizes = (int *)malloc(0);

    // this is where the keys are stored, corresponding to the 
    // rows of this->segments
    this->num_segment_keys = 0;
    this->segment_keys =(unsigned long long *) malloc(0);

}

// this is how you add a new segment
// key is the name of teh segment (i.e., "tokenseg")
// bio is the series of "BIO" markers corresponding to the raw text
bool Text::addSegmentBIO(char* key, char* bio) {
    
    char curr;

    // bio must have one character for each character in the raw text object
    if(strlen(bio) != (this->raw_size-1)) return false;

    int key_index = getKeyIndexAddIfNew(key,(int)this->raw_size-1);
    int * segment = this->segments[key_index];
    
    for(int i=0; i<this->raw_size-1; i++) {
        curr = bio[i];
        if(curr == 'B') segment[i] = 0;
        if(curr == 'I') segment[i] = 1;
        if(curr == 'O') segment[i] = 2;
    }
    return true;
}

// figure out which row of this->segment a key corresponds to
// if the key does not exist, add a new segment row and key
int Text::getKeyIndexAddIfNew(char* key,int segment_length) {
    unsigned long long keyHash = this->hashSegmentKey(key);
    int keyFoundAtIndex = -1;
    for(int i=0; i<this->num_segment_keys; i++) {
        if(this->segment_keys[i] == keyHash) keyFoundAtIndex = i;
    }
    if(keyFoundAtIndex != -1) {
        return keyFoundAtIndex;
    } else {
        return this->addSegmentAndKey(keyHash,segment_length);
    }
    return this->num_segment_keys - 1;
}

int Text::addSegmentAndKey(unsigned long long keyHash,int segment_length) {

    // adding key
    this->num_segment_keys += 1;
    this->segment_keys = (unsigned long long *)realloc(this->segment_keys,this->num_segment_keys * sizeof(unsigned long long));
    this->segment_keys[this->num_segment_keys - 1] = keyHash;

    // adding segment
    this->segments = (int **)realloc(this->segments,this->num_segment_keys*sizeof(int*));
    this->segment_buffer_sizes = (int *)realloc(this->segment_buffer_sizes,this->num_segment_keys*sizeof(int));
    this->segments[this->num_segment_keys - 1] = (int *)malloc(segment_length * sizeof(int));
    this->segment_buffer_sizes[this->num_segment_keys - 1] = segment_length; // initial size of 1000

    return this->num_segment_keys - 1;
}


/*
This hashes the segment key assuming that the key is all lowercase letters.
I assume that keys will be less than 13 letters, longer might cause collisions.
*/
unsigned long long Text::hashSegmentKey(char* key) {
    unsigned long keyHash = 0;
    char tmp;
    unsigned long long size = strlen(key);
    if(size > LONGESTKEY) {
      size = LONGESTKEY;
      std::cout << "WARNING: Key longer than 13 lowercase letter characters. Collision possible." <<  std::endl;
    } 

    for(int i=0; i<size; i++) {
        tmp = key[i];
        if(tmp > 96) {
            tmp -= 97;
        }
        tmp = tmp % 26;
        keyHash = (keyHash * 26) + tmp; // 26 letters
    }
    return keyHash;
}

Vocab::Vocab() {
    
    this->size = 0;

    this->vocab_buffer_size = 1000;
    this->vocab = (Term *) malloc(this->vocab_buffer_size * sizeof(Term));

    this->hash_size = 1000000;
    this->hash_table = (int *) malloc(this->hash_size * sizeof(int));
    for(int i=0; i<this->hash_size; i++) hash_table[i] = -1;

}

int Vocab::addTerm(char* term) {

    this->size += 1;
    
    Term new_term = this->vocab[this->size - 1];
    new_term.letters = (char*)malloc(strlen(term)+1);
    strcpy(new_term.letters, term);

    int hash = this->getTermHash(term);
    while(this->hash_table[hash] == -1) hash += 1;
    this->hash_table[hash] = this->size-1;

    return this->size-1;
}

int Vocab::getTermHash(char* term) {
    unsigned long long hash = 0;
    for(int i=0; i<strlen(term); i++) {
        hash = ((hash + term[i]) * CHAR_VOCAB_SIZE) % this->hash_size;
    }
    return (int) hash;
}

char* Text::getRaw() {
    return this->raw;
}

char* fact(char * text) {
    return text;
}
