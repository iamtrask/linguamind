/* File: linguamind.i */
%module linguamind

%{
#define SWIG_FILE_WITH_INIT
#include "linguamind.h"
%}

int fact(int n);