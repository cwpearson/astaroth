
Preprocessed Scalar
value(in Scalar vertex)
{
    return vertex[vertexIdx];
}

Preprocessed Vector
gradient(in Scalar vertex)
{
    return (Vector){derx(vertexIdx, vertex),
                    dery(vertexIdx, vertex),
                    derz(vertexIdx, vertex)};
}

Preprocessed Matrix
hessian(in Scalar vertex)
{
    Matrix hessian;

    hessian.row[0] = (Vector){derxx(vertexIdx, vertex), derxy(vertexIdx, vertex), derxz(vertexIdx, vertex)};
    hessian.row[1] = (Vector){hessian.row[0].y,       deryy(vertexIdx, vertex), deryz(vertexIdx, vertex)};
    hessian.row[2] = (Vector){hessian.row[0].z,       hessian.row[1].z,       derzz(vertexIdx, vertex)};

    return hessian;
}
