#include "vocab.h"
#include <assert.h>
#include <math.h>

Vocab::Vocab() {
    
    this->size = 0;

    this->vocab_buffer_size = 100000;
    this->vocab = (Term *) malloc(this->vocab_buffer_size * sizeof(Term));

    this->hash_size = 1000000;
    this->hash_table = (int *) malloc(this->hash_size * sizeof(int));
    for(int i=0; i<this->hash_size; i++) hash_table[i] = -1;

    this->addTerm("</s>");
}

Vocab::Vocab(Vocab* old) {
    
    this->size = old->size;

    this->vocab_buffer_size = old->vocab_buffer_size;
    this->vocab = (Term *) malloc(old->vocab_buffer_size * sizeof(Term));

    for(int i=0; i<old->size; i++) {
        this->vocab[i].count = old->vocab[i].count;
        this->vocab[i].letters = (char*)malloc(strlen(old->vocab[i].letters)+1);
        strcpy(this->vocab[i].letters,old->vocab[i].letters);
    }

    this->hash_size = old->hash_size;
    this->hash_table = (int *) malloc(old->hash_size * sizeof(int));
    for(int i=0; i<old->hash_size; i++) this->hash_table[i] = old->hash_table[i];
}

int Vocab::getTermIndex(char* term) {

    unsigned int hash = this->getTermHash(term);
   
    while (1) {
        if (this->hash_table[hash] == -1) return -1;
        if (!strcmp(term, this->vocab[this->hash_table[hash]].letters)) return this->hash_table[hash];
        hash = (hash + 1) % this->hash_size;
    }
 
    // term doesn't exist
    return -1;
}

int Vocab::getUnigramValue(int index) {
  if(this->unigram_table == NULL) {
    this->InitUnigramTable();
  }
  return this->unigram_table[index];
}

void Vocab::InitUnigramTable() {
 
  int a, i;
  double train_words_pow = 0;
  double d1, power = 0.75;
  this->unigram_table = (int *)malloc(this->unigram_table_size * sizeof(int));
  for (a = 0; a < this->size; a++) train_words_pow += pow(this->vocab[a].count, power);
  i = 0;
  d1 = pow(this->vocab[i].count, power) / train_words_pow;
  for (a = 0; a < this->unigram_table_size; a++) {
    this->unigram_table[a] = i;
    if (a / (double)this->unigram_table_size > d1) {
      i++;
      d1 += pow(this->vocab[i].count, power) / train_words_pow;
    }
    if (i >= this->size) i = this->size - 1;
  }

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

        // printf("Adding Term at Index:%i\n",this->size-1);

        return this->size-1;
    } else {
        return index;
    }
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
    return ((struct Term *)b)->count - ((struct Term *)a)->count;
}

// Sorts the vocabulary by frequency using word counts
void Vocab::sort(int min_count) {
  int a, size;
  unsigned int hash;
  // Sort the vocabulary and keep </s> at the first position
  qsort(&this->vocab[1], this->size - 1, sizeof(struct Term), VocabCompare);
  for (a = 0; a < this->hash_size; a++) this->hash_table[a] = -1;
  size = this->size;
  
  for (a = 0; a < size; a++) {
    // Words occuring less than min_count times will be discarded from the vocab
    if ((this->vocab[a].count < min_count) && (a != 0)) {
      free(this->vocab[a].letters);
      this->size--;
    } else {
      // Hash will be re-computed, as after the sorting it is not actual
      hash=this->getTermHash(this->vocab[a].letters);
      while (this->hash_table[hash] != -1) hash = (hash + 1) % this->hash_size;
      this->hash_table[hash] = a;

    }
  }
  this->vocab = (struct Term *)realloc(this->vocab, (this->size + 1) * sizeof(struct Term));

}


Term* Vocab::getTermAtIndex(int i) {
    return &this->vocab[i];
}

unsigned int Vocab::getTermHash(char* term) {
    unsigned long long hash = 0;
    for(int i=0; i<strlen(term); i++) {
        hash = ((hash + term[i]) * CHAR_VOCAB_SIZE) % this->hash_size;
    }
    return (unsigned int) hash;
}

void Vocab::createBinaryTree() {
  int redundancy = 1;
  int n_labels = this->size;
  
  Vector* ptr;

  int osz_ = n_labels * redundancy;
    int redundancy_ = redundancy;
    int n_labels_ = n_labels;

  tree.resize(2 * osz_ - 1);
  for (int32_t i = 0; i < 2 * osz_ - 1; i++) {
      tree[i].parent = -1;
      tree[i].left = -1;
      tree[i].right = -1;
      tree[i].count = 1e15;
      tree[i].binary = false;
      tree[i].output = -1;
    }

    int32_t leaf = osz_ - 1;
    int32_t node = osz_;
    
    for (int32_t i = osz_; i < 2 * osz_ - 1; i++) {
      int32_t mini[2];
      mini[0] = 0;
      mini[1] = 0;
      for (int32_t j = 0; j < 2; j++) {
        // if (leaf >= 0 && tree[leaf].count < tree[node].count) {
        if(leaf >= 0) {
          mini[j] = leaf--;
        } else {
          mini[j] = node++;
        }
      }
      tree[i].left = mini[0];
      tree[i].right = mini[1];
      tree[i].count = tree[mini[0]].count + tree[mini[1]].count;
      tree[mini[0]].parent = i;
      tree[mini[1]].parent = i;
      tree[mini[1]].binary = true;
    }
    for (int32_t i = 0; i < osz_; i++) {
      std::vector<int32_t> path;
      std::vector<bool> code;
      int32_t j = i;
      while (tree[j].parent != -1) {
        path.push_back(tree[j].parent - osz_);
        code.push_back(tree[j].binary);
        j = tree[j].parent;
      }
      paths.push_back(path);
      ptr = new Vector(code);
      codes.push_back(ptr);
      code.clear();
    }
}

int Vocab::getCodeSize(int i) {return this->codes[i]->size;}
int Vocab::getPathSize(int i) {return this->paths[i].size();}
std::vector<int> &Vocab::getPathReference(int i) {return this->paths[i];}
Vector* Vocab::getCodeReference(int i) {return this->codes[i];}
bool Vocab::getCode(int i, int j) {return this->codes[i]->get(i);}
int Vocab::getPath(int i, int j) {return this->paths[i][j];}

