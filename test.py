import linguamind.linalg as la
import linguamind.nn as nn

seed = la.Seed(1)

mat = la.Matrix(6,10)
mat.uniform(seed)
mat -= float(0.5)
mat /= float(10)

assert mat[0][0] == 0.04002685472369194
assert mat[5][9] == -0.006083679385483265