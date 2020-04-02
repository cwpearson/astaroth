#pragma once
#include "astaroth.h"

#if AC_MPI_ENABLED
#include <mpi.h>

#define AC_MPI_UNIDIRECTIONAL_COMM (0)
#endif // AC_MPI_ENABLED

typedef struct {
    int3 dims;
    AcReal* data;

#if (AC_MPI_ENABLED && AC_MPI_UNIDIRECTIONAL_COMM)
    MPI_Win win; // MPI window for RMA
#endif           // (AC_MPI_ENABLED && AC_MPI_UNIDIRECTIONAL_COMM)
} PackedData;

typedef struct {
    AcReal* in[NUM_VTXBUF_HANDLES];
    AcReal* out[NUM_VTXBUF_HANDLES];

    AcReal* profiles[NUM_SCALARARRAY_HANDLES];
} VertexBufferArray;

struct device_s {
    int id;
    AcMeshInfo local_config;

    // Concurrency
    cudaStream_t streams[NUM_STREAMS];

    // Memory
    VertexBufferArray vba;
    AcReal* reduce_scratchpad;
    AcReal* reduce_result;
};

#ifdef __cplusplus
extern "C" {
#endif

/** */
AcResult acKernelPeriodicBoundconds(const cudaStream_t stream, const int3 start, const int3 end,
                                    AcReal* vtxbuf);

/** */
AcResult acKernelDummy(void);

/** */
AcResult acKernelAutoOptimizeIntegration(const int3 start, const int3 end, VertexBufferArray vba);

/** */
AcResult acKernelIntegrateSubstep(const cudaStream_t stream, const int step_number,
                                  const int3 start, const int3 end, VertexBufferArray vba);

/** */
AcResult acKernelPackData(const cudaStream_t stream, const VertexBufferArray vba,
                          const int3 vba_start, PackedData packed);

/** */
AcResult acKernelUnpackData(const cudaStream_t stream, const PackedData packed,
                            const int3 vba_start, VertexBufferArray vba);

/** */
AcReal acKernelReduceScal(const cudaStream_t stream, const ReductionType rtype, const int3 start,
                          const int3 end, const AcReal* vtxbuf, AcReal* scratchpad,
                          AcReal* reduce_result);

/** */
AcReal acKernelReduceVec(const cudaStream_t stream, const ReductionType rtype, const int3 start,
                         const int3 end, const AcReal* vtxbuf0, const AcReal* vtxbuf1,
                         const AcReal* vtxbuf2, AcReal* scratchpad, AcReal* reduce_result);

#ifdef __cplusplus
} // extern "C"
#endif
