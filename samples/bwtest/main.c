#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>

#include <cuda_runtime_api.h>

#include "timer_hires.h" // From src/common

//#define BLOCK_SIZE (100 * 1024 * 1024) // Bytes
#define BLOCK_SIZE (256 * 256 * 3 * 8 * 8)

/*
  Findings:
    - MUST ALWAYS SET DEVICE. Absolutely kills performance if device is not set explicitly
    - Need to use cudaMalloc for intranode comm for P2P to trigger with MPI
    - For internode one should use pinned memory (RDMA is staged through pinned, gives full
    network speed iff pinned)
*/

static uint8_t*
allocHost(const size_t bytes)
{
    uint8_t* arr = malloc(bytes);
    assert(arr);
    return arr;
}

static void
freeHost(uint8_t* arr)
{
    free(arr);
}

static uint8_t*
allocDevice(const size_t bytes)
{
    uint8_t* arr;
    // Standard (20 GiB/s internode, 85 GiB/s intranode)
    // const cudaError_t retval = cudaMalloc((void**)&arr, bytes);
    // Unified mem (5 GiB/s internode, 6 GiB/s intranode)
    // const cudaError_t retval = cudaMallocManaged((void**)&arr, bytes, cudaMemAttachGlobal);
    // Pinned (40 GiB/s internode, 10 GiB/s intranode)
    const cudaError_t retval = cudaMallocHost((void**)&arr, bytes);
    assert(retval == cudaSuccess);
    return arr;
}

static void
freeDevice(uint8_t* arr)
{
    cudaFree(arr);
}

static void
sendrecv_blocking(uint8_t* src, uint8_t* dst)
{
    int pid, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    int nfront = (pid + 1) % nprocs;
    int nback  = (((pid - 1) % nprocs) + nprocs) % nprocs;

    if (!pid) {
        MPI_Status status;
        MPI_Send(src, BLOCK_SIZE, MPI_BYTE, nfront, pid, MPI_COMM_WORLD);
        MPI_Recv(dst, BLOCK_SIZE, MPI_BYTE, nback, nback, MPI_COMM_WORLD, &status);
    }
    else {
        MPI_Status status;
        MPI_Recv(dst, BLOCK_SIZE, MPI_BYTE, nback, nback, MPI_COMM_WORLD, &status);
        MPI_Send(src, BLOCK_SIZE, MPI_BYTE, nfront, pid, MPI_COMM_WORLD);
    }
}

static void
sendrecv_nonblocking(uint8_t* src, uint8_t* dst)
{
    int pid, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    int nfront = (pid + 1) % nprocs;
    int nback  = (((pid - 1) % nprocs) + nprocs) % nprocs;

    MPI_Request recv_request, send_request;
    MPI_Irecv(dst, BLOCK_SIZE, MPI_BYTE, nback, nback, MPI_COMM_WORLD, &recv_request);
    MPI_Isend(src, BLOCK_SIZE, MPI_BYTE, nfront, pid, MPI_COMM_WORLD, &send_request);

    MPI_Status status;
    MPI_Wait(&recv_request, &status);
    MPI_Wait(&send_request, &status);
}

static void
sendrecv_twoway(uint8_t* src, uint8_t* dst)
{
    int pid, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    int nfront = (pid + 1) % nprocs;
    int nback  = (((pid - 1) % nprocs) + nprocs) % nprocs;

    MPI_Status status;
    MPI_Sendrecv(src, BLOCK_SIZE, MPI_BYTE, nfront, pid, dst, BLOCK_SIZE, MPI_BYTE, nback, nback,
                 MPI_COMM_WORLD, &status);
}

#define PRINT                                                                                      \
    if (!pid)                                                                                      \
    printf

static void
measurebw(const char* msg, const size_t bytes, void (*sendrecv)(uint8_t*, uint8_t*), uint8_t* src,
          uint8_t* dst)
{
    const size_t num_samples = 10;

    int pid, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    PRINT("%s\n", msg);
    MPI_Barrier(MPI_COMM_WORLD);

    PRINT("\tWarming up... ");
    for (size_t i = 0; i < num_samples / 10; ++i)
        sendrecv(src, dst);

    MPI_Barrier(MPI_COMM_WORLD);
    PRINT("Done\n");

    PRINT("\tBandwidth... ");
    fflush(stdout);

    Timer t;
    MPI_Barrier(MPI_COMM_WORLD);
    timer_reset(&t);
    MPI_Barrier(MPI_COMM_WORLD);

    for (size_t i = 0; i < num_samples; ++i)
        sendrecv(src, dst);

    MPI_Barrier(MPI_COMM_WORLD);
    const long double time_elapsed = timer_diff_nsec(t) / 1e9l; // seconds
    PRINT("%Lg GiB/s\n", num_samples * bytes / time_elapsed / (1024 * 1024 * 1024));
    PRINT("\tTransfer time: %Lg ms\n", time_elapsed * 1000);
    MPI_Barrier(MPI_COMM_WORLD);
}

int
main(void)
{
    // Disable stdout buffering
    setbuf(stdout, NULL);

    MPI_Init(NULL, NULL);

    int pid, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    assert(nprocs >= 2); // Require at least one neighbor

    int devices_per_node = -1;
    cudaGetDeviceCount(&devices_per_node);
    const int device_id = pid % devices_per_node;
    cudaSetDevice(device_id);

    printf("Process %d of %d running.\n", pid, nprocs);
    MPI_Barrier(MPI_COMM_WORLD);

    PRINT("Block size: %u MiB\n", BLOCK_SIZE / (1024 * 1024));

    {
        uint8_t* src = allocHost(BLOCK_SIZE);
        uint8_t* dst = allocHost(BLOCK_SIZE);

        measurebw("Unidirectional bandwidth, blocking (Host)", //
                  2 * BLOCK_SIZE, sendrecv_blocking, src, dst);
        measurebw("Bidirectional bandwidth, async (Host)", //
                  2 * BLOCK_SIZE, sendrecv_nonblocking, src, dst);
        measurebw("Bidirectional bandwidth, twoway (Host)", //
                  2 * BLOCK_SIZE, sendrecv_twoway, src, dst);

        freeHost(src);
        freeHost(dst);
    }

    {
        uint8_t* src = allocDevice(BLOCK_SIZE);
        uint8_t* dst = allocDevice(BLOCK_SIZE);

        measurebw("Unidirectional bandwidth, blocking (Device)", //
                  2 * BLOCK_SIZE, sendrecv_blocking, src, dst);
        measurebw("Bidirectional bandwidth, async (Device)", //
                  2 * BLOCK_SIZE, sendrecv_nonblocking, src, dst);
        measurebw("Bidirectional bandwidth, twoway (Device)", //
                  2 * BLOCK_SIZE, sendrecv_twoway, src, dst);

        freeDevice(src);
        freeDevice(dst);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
