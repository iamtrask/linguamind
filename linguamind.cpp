#include "linguamind.h"
#include "test.h"
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
        if(curr == 'B') {
            segment[i] = 0;
        } else if(curr == 'I')  {
            segment[i] = 1;
        } else {
            segment[i] = 2;
        }
    }


    return true;
}

// takes a segment and rolls it up into a new string of objects
// for example, character level BIO would roll up into (an array of) unique tokens (ids)
// which subsequently creates a new vocabulary for token ids as well
std::vector<int> Text::rollupSegmentNewVocab(char* key) {

    Vocab* vocab = new Vocab();
    int segment_index = this->getKeyIndexAddIfNew(key,-1);

    int sequence_buffer_length = this->segment_buffer_sizes[segment_index];
    int* sequence = (int*)malloc(sequence_buffer_length * sizeof(int));
    int sequence_length = 0;

    int * segment = this->segments[segment_index];

    char * term;
    int term_index;

    int start = -1;
    int end = -1;
    for(int i=0; i<this->segment_buffer_sizes[segment_index]; i++) {
        if(start == -1) {
            if(segment[i] == 0) {
              start = i;  
            } 
        } else {
            if(segment[i] == 0) {
                end = i;

                // finish word
                term = (char*)malloc(end - start);
                for(int j=start; j < end; j++) term[j-start] = j;
                term_index = vocab->addTerm(term);
                free(term);
                sequence[sequence_length] = term_index;
                sequence_length++;

                if(sequence_length > sequence_buffer_length - 2) {
                    sequence_buffer_length *= 2;
                    sequence = (int*)realloc(sequence,sequence_buffer_length*sizeof(int));
                }

                // start another
                start = i;
            } else if(segment[i] == 1) {

            } else if(segment[i] == 2) {
                end = i-1;

                // finish word
                term = (char*)malloc(end - start);
                for(int j=start; j < end; j++) term[j-start] = j;
                term_index = vocab->addTerm(term);
                free(term);
                sequence[sequence_length] = term_index;
                sequence_length++;

                if(sequence_length > sequence_buffer_length - 2) {
                    sequence_buffer_length *= 2;
                    sequence = (int*)realloc(sequence,sequence_buffer_length*sizeof(int));
                }

                start = -1;
            }
        }
    }

    if(start != -1) {
        end = this->segment_buffer_sizes[segment_index]-1;

        // finish word
        term = (char*)malloc(end - start);
        for(int j=start; j < end; j++) term[j-start] = j;
        term_index = vocab->addTerm(term);
        free(term);
        sequence[sequence_length] = term_index;
        sequence_length++;

        if(sequence_length > sequence_buffer_length - 2) {
            sequence_buffer_length *= 2;
            sequence = (int*)realloc(sequence,sequence_buffer_length*sizeof(int));
        }
    }

    std::vector<int> v;
    for (int i=0; i<this->segment_buffer_sizes[segment_index]; i++) {
        v.push_back(segment[i]);
    }
    v.push_back(-1);
    for(int i=0; i<sequence_length; i++) {
        v.push_back(sequence[i]);
    }

    // int[] sequence_array = new int[sequence_length];
    // for(int i=0; i<sequence_length; i++) {
    //     sequence_array[i] = sequence[i];
    // }
    // free(sequence);

    return v;
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

char* Text::getRaw() {
    return this->raw;
}

char* fact(char * text) {
    return text;
}
