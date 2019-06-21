
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

Preprocessed Scalar
der6x_upwd(in Scalar vertex), 
{
    return (Scalar){(1.0/60.0)*inv_ds* (
                    - 20.0* vertex[vertexIdx.x,   vertexIdx.y, vertexIdx.z] 
                    + 15.0*(vertex[vertexIdx.x, vertexIdx.y+1, vertexIdx.z] + vertex[vertexIdx.x, vertexIdx.y-1, vertexIdx.z]) 
                    -  6.0*(vertex[vertexIdx.x, vertexIdx.y+2, vertexIdx.z] + vertex[vertexIdx.x, vertexIdx.y-2, vertexIdx.z])
                    +       vertex[vertexIdx.x, vertexIdx.y+3, vertexIdx.z] + vertex[vertexIdx.x, vertexIdx.y-3, vertexIdx.z];
                    )}
}

Preprocessed Scalar
der6y_upwd(in Scalar vertex), 
{
    return (Scalar){(1.0/60.0)*inv_ds* (
                    - 20.0* vertex[vertexIdx.x,   vertexIdx.y, vertexIdx.z] 
                    + 15.0*(vertex[vertexIdx.x, vertexIdx.y+1, vertexIdx.z] + vertex[vertexIdx.x, vertexIdx.y-1, vertexIdx.z]) 
                    -  6.0*(vertex[vertexIdx.x, vertexIdx.y+2, vertexIdx.z] + vertex[vertexIdx.x, vertexIdx.y-2, vertexIdx.z])
                    +       vertex[vertexIdx.x, vertexIdx.y+3, vertexIdx.z] + vertex[vertexIdx.x, vertexIdx.y-3, vertexIdx.z];
                    )}
}

Preprocessed Scalar
der6z_upwd(in Scalar vertex), 
{
    return (Scalar){(1.0/60.0)*inv_ds* (
                    - 20.0* vertex[vertexIdx.x,   vertexIdx.y, vertexIdx.z] 
                    + 15.0*(vertex[vertexIdx.x, vertexIdx.y, vertexIdx.z+1] + vertex[vertexIdx.x, vertexIdx.y, vertexIdx.z-1]) 
                    -  6.0*(vertex[vertexIdx.x, vertexIdx.y, vertexIdx.z+2] + vertex[vertexIdx.x, vertexIdx.y, vertexIdx.z-2])
                    +       vertex[vertexIdx.x, vertexIdx.y, vertexIdx.z+3] + vertex[vertexIdx.x, vertexIdx.y, vertexIdx.z-3];
                    )}
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
