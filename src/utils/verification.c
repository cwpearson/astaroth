#include "astaroth_utils.h"

#include <math.h>
#include <stdbool.h>
#include <string.h>

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

static inline bool
is_valid(const AcReal a)
{
    return !isnan(a) && !isinf(a);
}

Error
acGetError(AcReal model, AcReal candidate)
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

// Get the maximum absolute error. Works well if all the values in the mesh are approximately
// in the same range.
// Finding the maximum ulp error is not useful, as it picks up on the noise beyond the
// floating-point precision range and gives huge errors with values that should be considered
// zero (f.ex. 1e-19 and 1e-22 give error of around 1e4 ulps)
static Error
get_max_abs_error(const VertexBufferHandle vtxbuf_handle, const AcMesh model_mesh,
                  const AcMesh candidate_mesh)
{
    AcReal* model_vtxbuf     = model_mesh.vertex_buffer[vtxbuf_handle];
    AcReal* candidate_vtxbuf = candidate_mesh.vertex_buffer[vtxbuf_handle];

    Error error;
    error.abs_error = -1;

    for (size_t i = 0; i < acVertexBufferSize(model_mesh.info); ++i) {

        Error curr_error = acGetError(model_vtxbuf[i], candidate_vtxbuf[i]);

        if (curr_error.abs_error > error.abs_error)
            error = curr_error;
    }

    error.handle = vtxbuf_handle;
    strncpy(error.label, vtxbuf_names[vtxbuf_handle], ERROR_LABEL_LENGTH - 1);
    error.label[ERROR_LABEL_LENGTH - 1] = '\0';
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
    // Accept the error if the relative error is < max_ulp_error ulps.
    // Also consider the error zero if it is less than the minimum value in the mesh scaled to
    // machine epsilon
    const long double max_ulp_error = 5;

    if (error.ulp_error < max_ulp_error)
        return true;
    else if (error.abs_error < error.minimum_magnitude * AC_REAL_EPSILON)
        return true;
    else
        return false;
}

bool
printErrorToScreen(const Error error)
{
    bool errors_found = false;

    printf("\t%-15s... ", error.label);
    if (is_acceptable(error)) {
        printf(GRN "OK! " RESET);
    }
    else {
        printf(RED "FAIL! " RESET);
        errors_found = true;
    }

    fprintf(stdout, "| %.3Lg (abs), %.3Lg (ulps), %.3Lg (rel). Range: [%.3g, %.3g]\n", //
            error.abs_error, error.ulp_error, error.rel_error,                         //
            (double)error.minimum_magnitude, (double)error.maximum_magnitude);

    return errors_found;
}

/** Returns true when successful, false if errors were found. */
AcResult
acVerifyMesh(const AcMesh model, const AcMesh candidate)
{
    printf("Errors at the point of the maximum absolute error:\n");

    bool errors_found = false;
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        Error field_error = get_max_abs_error(i, model, candidate);
        errors_found |= printErrorToScreen(field_error);
    }

    printf("%s\n", errors_found ? "Failure. Found errors in one or more vertex buffers"
                                : "Success. No errors found.");

    return errors_found ? AC_FAILURE : AC_SUCCESS;
}

/** Verification function for scalar reductions*/
AcResult
acVerifyScalReductions(const AcMesh model, const AcScalReductionTestCase* testCases,
                       const size_t numCases)
{
    printf("\nTesting scalar reductions:\n");

    bool errors_found = false;
    for (size_t i = 0; i < numCases; i++) {
        AcReal model_reduction = acModelReduceScal(model, testCases[i].rtype, testCases[i].vtxbuf);
        Error error            = acGetError(model_reduction, testCases[i].candidate);
        strncpy(error.label, testCases[i].label, ERROR_LABEL_LENGTH - 1);
        error.label[ERROR_LABEL_LENGTH - 1] = '\0';
        errors_found |= printErrorToScreen(error);
    }
    printf("%s\n", errors_found ? "Failure. Found errors in one or more scalar reductions"
                                : "Success. No errors found.");

    return errors_found ? AC_FAILURE : AC_SUCCESS;
}

/** Verification function for vector reductions*/
AcResult
acVerifyVecReductions(const AcMesh model, const AcVecReductionTestCase* testCases,
                      const size_t numCases)
{
    printf("\nTesting vector reductions:\n");

    bool errors_found = false;
    for (size_t i = 0; i < numCases; i++) {
        AcReal model_reduction = acModelReduceVec(model, testCases[i].rtype, testCases[i].a,
                                                  testCases[i].b, testCases[i].c);
        Error error            = acGetError(model_reduction, testCases[i].candidate);
        strncpy(error.label, testCases[i].label, ERROR_LABEL_LENGTH - 1);
        error.label[ERROR_LABEL_LENGTH - 1] = '\0';
        errors_found |= printErrorToScreen(error);
    }
    printf("%s\n", errors_found ? "Failure. Found errors in one or more vector reductions"
                                : "Success. No errors found.");

    return errors_found ? AC_FAILURE : AC_SUCCESS;
}

/** Constructor for scalar reduction test case */
AcScalReductionTestCase
acCreateScalReductionTestCase(const char* label, const VertexBufferHandle vtxbuf, const ReductionType rtype)
{
    AcScalReductionTestCase testCase;

    strncpy(testCase.label,label,ERROR_LABEL_LENGTH - 1);
    testCase.label[ERROR_LABEL_LENGTH - 1] = '\0';
    testCase.vtxbuf = vtxbuf;
    testCase.rtype = rtype;
    testCase.candidate = 0;

    return testCase;
}

/** Constructor for vector reduction test case */
AcVecReductionTestCase
acCreateVecReductionTestCase(const char* label, const VertexBufferHandle a,
                const VertexBufferHandle b, const VertexBufferHandle c, const ReductionType rtype)
{
    AcVecReductionTestCase testCase;

    strncpy(testCase.label,label,ERROR_LABEL_LENGTH - 1);
    testCase.label[ERROR_LABEL_LENGTH - 1] = '\0';
    testCase.a = a;
    testCase.b = b;
    testCase.c = c;
    testCase.rtype = rtype;
    testCase.candidate = 0;

    return testCase;
}
