#include "text.h"
#include <assert.h>

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