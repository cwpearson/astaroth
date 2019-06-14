// TODO comments and reformatting

//Scalar
//dostuff(in Scalar uux)
//{
//   return uux[vertexIdx.x, vertexIdx.y, vertexIdx.z];
//}

// stencil_assembly.in
Preprocessed Scalar
some_exotic_stencil_computation(in Scalar uux)
{
    //#if STENCIL_ORDER == 2
    //    const Scalar coefficients[] = {1, 1, 1};
    //#else if STENCIL_ORDER == 4
    //    const Scalar coefficients[] = {....};
    //#endif

    int i = vertexIdx.x;
    int j = vertexIdx.y;
    int k = vertexIdx.z;
    const Scalar coefficients[] = {1, 2, 3};

    return coefficients[0] * uux[i-1, j, k] + 
           coefficients[1] * uux[i, j, k] + 
           coefficients[2] * uux[i+1, j, k];
}

// stencil_process.in
//in Scalar uux_in = VTXBUF_UUX;
//out Scalar uux_out = VTXBUF_UUX;


//Kernel
//solve(Scalar dt)
//{
//    uux_out = some_exotic_stencil(uux_in);
//}











