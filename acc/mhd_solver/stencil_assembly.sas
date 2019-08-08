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

#if LUPWD

Preprocessed Scalar
der6x_upwd(in Scalar vertex)
{
    Scalar inv_ds = DCONST_REAL(AC_inv_dsx);

    return (Scalar){  Scalar(1.0/60.0)*inv_ds* (
                    - Scalar(20.0)* vertex[vertexIdx.x,   vertexIdx.y, vertexIdx.z]
                    + Scalar(15.0)*(vertex[vertexIdx.x+1, vertexIdx.y, vertexIdx.z] 
                    +               vertex[vertexIdx.x-1, vertexIdx.y, vertexIdx.z])
                    - Scalar( 6.0)*(vertex[vertexIdx.x+2, vertexIdx.y, vertexIdx.z] 
                    +               vertex[vertexIdx.x-2, vertexIdx.y, vertexIdx.z])
                    +               vertex[vertexIdx.x+3, vertexIdx.y, vertexIdx.z] 
                    +               vertex[vertexIdx.x-3, vertexIdx.y, vertexIdx.z])};
}

Preprocessed Scalar
der6y_upwd(in Scalar vertex)
{
    Scalar inv_ds = DCONST_REAL(AC_inv_dsy);

    return (Scalar){ Scalar(1.0/60.0)*inv_ds* (
                    -Scalar( 20.0)* vertex[vertexIdx.x,   vertexIdx.y, vertexIdx.z]
                    +Scalar( 15.0)*(vertex[vertexIdx.x, vertexIdx.y+1, vertexIdx.z] 
                    +               vertex[vertexIdx.x, vertexIdx.y-1, vertexIdx.z])
                    -Scalar(  6.0)*(vertex[vertexIdx.x, vertexIdx.y+2, vertexIdx.z] 
                    +               vertex[vertexIdx.x, vertexIdx.y-2, vertexIdx.z])
                    +               vertex[vertexIdx.x, vertexIdx.y+3, vertexIdx.z] 
                    +               vertex[vertexIdx.x, vertexIdx.y-3, vertexIdx.z])};
}

Preprocessed Scalar
der6z_upwd(in Scalar vertex)
{
    Scalar inv_ds = DCONST_REAL(AC_inv_dsz);

    return (Scalar){ Scalar(1.0/60.0)*inv_ds* (
                    -Scalar( 20.0)* vertex[vertexIdx.x,   vertexIdx.y, vertexIdx.z]
                    +Scalar( 15.0)*(vertex[vertexIdx.x, vertexIdx.y, vertexIdx.z+1] 
                    +               vertex[vertexIdx.x, vertexIdx.y, vertexIdx.z-1])
                    -Scalar(  6.0)*(vertex[vertexIdx.x, vertexIdx.y, vertexIdx.z+2] 
                    +               vertex[vertexIdx.x, vertexIdx.y, vertexIdx.z-2])
                    +               vertex[vertexIdx.x, vertexIdx.y, vertexIdx.z+3] 
                    +               vertex[vertexIdx.x, vertexIdx.y, vertexIdx.z-3])};
}

#endif

Preprocessed Matrix
hessian(in Scalar vertex)
{
    Matrix hessian;

    hessian.row[0] = (Vector){derxx(vertexIdx, vertex), derxy(vertexIdx, vertex), derxz(vertexIdx, vertex)};
    hessian.row[1] = (Vector){hessian.row[0].y,       deryy(vertexIdx, vertex), deryz(vertexIdx, vertex)};
    hessian.row[2] = (Vector){hessian.row[0].z,       hessian.row[1].z,       derzz(vertexIdx, vertex)};

    return hessian;
}
