#ifndef TEXT
#define TEXT

#include <iostream>
#include <vector>

#include "vocab.h"

#define LONGESTKEY 13 // 9,223,372,036,854,775,807 - (26**13)
#define CHAR_VOCAB_SIZE 256

class Text {
	public:
		Text(char* filepath);
};

#endif