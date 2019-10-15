#include "verification.h"

#include <math.h>
#include <stdio.h>

#include "astaroth.h"

#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

#define fabs(x) ((_Generic((x), float : fabsf, double : fabs, long double : fabsl))(x))

// Defines for colored output
#define RED "\x1B[31m"
#define GRN "\x1B[32m"
#define YEL "\x1B[33m"
#define BLU "\x1B[34m"
#define MAG "\x1B[35m"
#define CYN "\x1B[36m"
#define WHT "\x1B[37m"
#define RESET "\x1B[0m"

typedef struct {
    VertexBufferHandle handle;
    AcReal model;
    AcReal candidate;
    long double abs_error;
    long double ulp_error;
    long double rel_error;
    AcReal maximum_magnitude;
    AcReal minimum_magnitude;
} Error;

static inline bool
is_valid(const AcReal a)
{
    return !isnan(a) && !isinf(a);
}

static Error
get_error(AcReal model, AcReal candidate)
{
    Error error;
    error.abs_error = 0;

    error.model     = model;
    error.candidate = candidate;

    if (error.model == error.candidate || fabsl(model - candidate) == 0) { // If exact
        error.abs_error = 0;
        error.rel_error = 0;
        error.ulp_error = 0;
    }
    else if (!is_valid(error.model) || !is_valid(error.candidate)) {
        error.abs_error = INFINITY;
        error.rel_error = INFINITY;
        error.ulp_error = INFINITY;
    }
    else {
        const int base = 2;
        const int p    = sizeof(AcReal) == 4 ? 24 : 53; // Bits in the significant

        const long double e = floorl(logl(fabsl(error.model)) / logl(2));

        const long double ulp             = powl(base, e - (p - 1));
        const long double machine_epsilon = 0.5 * powl(base, -(p - 1));
        error.abs_error                   = fabsl(model - candidate);
        error.ulp_error                   = error.abs_error / ulp;
        error.rel_error                   = fabsl(1.0l - candidate / model) / machine_epsilon;
    }

    return error;
}

static AcReal
get_maximum_magnitude(const AcReal* field, const AcMeshInfo info)
{
    AcReal maximum = -INFINITY;

    for (size_t i = 0; i < acVertexBufferSize(info); ++i)
        maximum = max(maximum, fabs(field[i]));

    return maximum;
}

static AcReal
get_minimum_magnitude(const AcReal* field, const AcMeshInfo info)
{
    AcReal minimum = INFINITY;

    for (size_t i = 0; i < acVertexBufferSize(info); ++i)
        minimum = min(minimum, fabs(field[i]));

    return minimum;
}

static Error
get_max_abs_error(const VertexBufferHandle vtxbuf_handle, const AcMesh model_mesh,
                  const AcMesh candidate_mesh)
{
    AcReal* model_vtxbuf     = model_mesh.vertex_buffer[vtxbuf_handle];
    AcReal* candidate_vtxbuf = candidate_mesh.vertex_buffer[vtxbuf_handle];

    Error error;
    error.abs_error = -1;

    for (size_t i = 0; i < acVertexBufferSize(model_mesh.info); ++i) {

        Error curr_error = get_error(model_vtxbuf[i], candidate_vtxbuf[i]);

        if (curr_error.abs_error > error.abs_error)
            error = curr_error;
    }

    error.handle            = vtxbuf_handle;
    error.maximum_magnitude = get_maximum_magnitude(model_vtxbuf, model_mesh.info);
    error.minimum_magnitude = get_minimum_magnitude(model_vtxbuf, model_mesh.info);

    return error;
}

static inline void
print_error_to_file(const char* path, const int n, const Error error)
{
    FILE* file = fopen(path, "a");
    fprintf(file, "%d, %Lg, %Lg, %Lg, %g, %g\n", n, error.ulp_error, error.abs_error,
            error.rel_error, (double)error.maximum_magnitude, (double)error.minimum_magnitude);
    fclose(file);
}

static bool
is_acceptable(const Error error)
{
    // TODO FIXME
    const AcReal range = error.maximum_magnitude - error.minimum_magnitude;
    if (error.abs_error < range * AC_REAL_EPSILON)
        return true;
    else
        return false;
}

static void
print_error_to_screen(const Error error)
{
    printf("\t%-15s... ", vtxbuf_names[error.handle]);
    if (is_acceptable(error)) {
        printf(GRN "OK! " RESET);
    }
    else {
        printf(RED "FAIL! " RESET);
    }

    fprintf(stdout, "| %.2Lg (abs), %.2Lg (ulps), %.2Lg (rel). Range: [%.2g, %.2g]\n", //
            error.abs_error, error.ulp_error, error.rel_error,                         //
            (double)error.minimum_magnitude, (double)error.maximum_magnitude);
}

bool
acVerifyMesh(const AcMesh model, const AcMesh candidate)
{
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        Error field_error = get_max_abs_error(i, model, candidate);
        print_error_to_screen(field_error);
    }
    printf("WARNING: is_acceptable() not yet complete\n");
    return true;
}
