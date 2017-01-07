#include "seed.h"
#include <stdexcept> 

Seed::Seed(unsigned long long seed) {
	this->seed = seed;
}

void Seed::eat() {
	throw std::runtime_error("Looks like you ate some poisonous seed!!!");
}