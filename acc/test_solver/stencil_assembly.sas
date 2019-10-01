#include "stencil_definition.sdh"


//JP NOTE IMPORTANT/////////////////////////////////////////////////////////////////////////////////
// These functions are defined here temporarily.
//
// Currently the built-in functions (derx, derxx etc) are defined in CUDA in integrate.cuh.
// This is bad. Instead the built-in functions should be defined in the DSL, and be "includable"
// as a standard DSL library, analogous to f.ex. stdlib.h in C.
////////////////////////////////////////////////////////////////////////////////////////////////////


Preprocessed Scalar
value(in ScalarField vertex)
{
    return vertex[vertexIdx];
}

Preprocessed Vector
gradient(in ScalarField vertex)
{
    return (Vector){derx(vertexIdx, vertex), dery(vertexIdx, vertex), derz(vertexIdx, vertex)};
}

Preprocessed Matrix
hessian(in ScalarField vertex)
{
    Matrix hessian;

    hessian.row[0] = (Vector){derxx(vertexIdx, vertex), derxy(vertexIdx, vertex),
                              derxz(vertexIdx, vertex)};
    hessian.row[1] = (Vector){hessian.row[0].y, deryy(vertexIdx, vertex), deryz(vertexIdx, vertex)};
    hessian.row[2] = (Vector){hessian.row[0].z, hessian.row[1].z, derzz(vertexIdx, vertex)};

    return hessian;
}
