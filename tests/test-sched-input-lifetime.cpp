// ggml_backend_sched_compute_splits promises the caller may overwrite a tensor
// flagged GGML_TENSOR_FLAG_INPUT once a graph has been submitted:
//
//     // inputs from the user must be copied immediately to prevent the user
//     // overwriting the data before the copy is done
//
// It keeps that promise by copying the input, but only copies inputs that must
// move to another backend. An input already in a buffer the compute backend
// accepts is never copied, so nothing protects it - and the kernel dereferences
// it at execution time, not at submit time. Integrated GPUs take that path,
// because they accept their own pinned host buffer directly.
//
// Each case scatters into a destination through a reused index tensor and checks
// every row was written by the chunk that owns it.
//
//   shared      input in the backend's pinned host buffer, no synchronize
//   sync        same, synchronizing each submission
//   cpu-input   input in a plain CPU buffer the backend does not accept, so the
//               scheduler is forced to copy it - what a discrete GPU always does

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <vector>

static const int64_t W     = 32;    // destination row width
static const int64_t CHUNK = 512;   // rows written per submission
static const int64_t ITERS = 8;     // submissions
static const int64_t HEAVY = 2048;  // matmul size, to keep the queue deep

enum input_mode { INPUT_MODE_SHARED, INPUT_MODE_SYNC, INPUT_MODE_CPU_INPUT };

static int64_t run_case(ggml_backend_dev_t dev, ggml_backend_buffer_type_t host_buft, input_mode mode) {
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    const int64_t n_rows = CHUNK*ITERS;

    ggml_init_params ip = { ggml_tensor_overhead()*8, nullptr, true };

    ggml_context * ctx_d = ggml_init(ip);
    ggml_tensor * dst = ggml_new_tensor_2d(ctx_d, GGML_TYPE_F16, W, n_rows);
    ggml_tensor * ha  = ggml_new_tensor_2d(ctx_d, GGML_TYPE_F32, HEAVY, HEAVY);
    ggml_tensor * hb  = ggml_new_tensor_2d(ctx_d, GGML_TYPE_F32, HEAVY, CHUNK);
    ggml_backend_buffer_t buf_d = ggml_backend_alloc_ctx_tensors(ctx_d, backend);
    ggml_backend_buffer_clear(buf_d, 0);

    ggml_context * ctx_i = ggml_init(ip);
    ggml_tensor * idxs = ggml_new_tensor_1d(ctx_i, GGML_TYPE_I64, CHUNK);
    ggml_tensor * src  = ggml_new_tensor_2d(ctx_i, GGML_TYPE_F32, W, CHUNK);
    ggml_set_input(idxs);
    ggml_set_input(src);
    ggml_backend_buffer_t buf_i = ggml_backend_alloc_ctx_tensors_from_buft(
        ctx_i, mode == INPUT_MODE_CPU_INPUT ? ggml_backend_cpu_buffer_type() : host_buft);

    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_backend_t backends[2] = { backend, cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, 2048, false, true);

    std::vector<float>   sbuf(W*CHUNK);
    std::vector<int64_t> ibuf(CHUNK);

    for (int64_t it = 0; it < ITERS; ++it) {
        // stamp each chunk so a row written by the wrong chunk is distinguishable
        for (int64_t r = 0; r < CHUNK; ++r) {
            ibuf[r] = it*CHUNK + r;
            for (int64_t e = 0; e < W; ++e) sbuf[r*W + e] = (float) (it + 1);
        }
        // the unsafe write: a previous submission may still be reading these
        ggml_backend_tensor_set(idxs, ibuf.data(), 0, ggml_nbytes(idxs));
        ggml_backend_tensor_set(src,  sbuf.data(), 0, ggml_nbytes(src));

        ggml_init_params gp = { ggml_tensor_overhead()*16 + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(gp);
        ggml_cgraph  * gf  = ggml_new_graph(ctx);

        ggml_tensor * work = ggml_mul_mat(ctx, ha, hb);   // keep the queue deep
        for (int k = 0; k < 3; ++k) work = ggml_mul_mat(ctx, ha, work);
        ggml_build_forward_expand(gf, work);
        ggml_build_forward_expand(gf, ggml_set_rows(ctx, dst, src, idxs));

        ggml_backend_sched_reset(sched);
        GGML_ASSERT(ggml_backend_sched_alloc_graph(sched, gf));
        ggml_backend_sched_graph_compute_async(sched, gf);
        if (mode == INPUT_MODE_SYNC) {
            ggml_backend_sched_synchronize(sched);
        }
        ggml_free(ctx);
    }
    ggml_backend_sched_synchronize(sched);

    std::vector<uint16_t> got(W*n_rows);
    ggml_backend_tensor_get(dst, got.data(), 0, ggml_nbytes(dst));

    int64_t bad = 0;
    for (int64_t row = 0; row < n_rows; ++row) {
        bad += ggml_fp16_to_fp32(got[row*W]) != (float) (row/CHUNK + 1);
    }

    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(buf_i);
    ggml_backend_buffer_free(buf_d);
    ggml_free(ctx_i);
    ggml_free(ctx_d);
    ggml_backend_free(cpu);
    ggml_backend_free(backend);
    return bad;
}

int main() {
    const char * mode_name[] = { "shared", "sync", "cpu-input" };
    const input_mode modes[] = { INPUT_MODE_SHARED, INPUT_MODE_SYNC, INPUT_MODE_CPU_INPUT };

    size_t n_tested = 0;
    int    n_fail   = 0;

    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            continue;
        }

        // only backends that read a host buffer in place are exposed to this;
        // anything else is already protected by the copy the scheduler inserts
        ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(dev);
        if (!host_buft || !ggml_backend_dev_supports_buft(dev, host_buft)) {
            printf("%s: copies host inputs already, skipping\n", ggml_backend_dev_name(dev));
            continue;
        }

        for (size_t m = 0; m < 3; ++m) {
            printf("  %s / %-9s ", ggml_backend_dev_name(dev), mode_name[m]);
            fflush(stdout);
            const int64_t bad = run_case(dev, host_buft, modes[m]);
            if (bad) {
                printf("FAILED: %lld of %lld rows written by the wrong chunk\n",
                       (long long) bad, (long long) (CHUNK*ITERS));
                n_fail++;
            } else {
                printf("PASSED\n");
            }
        }
        n_tested++;
    }

    if (n_tested == 0) {
        printf("no backend reads host inputs in place, nothing to test\n");
        return 0;
    }
    if (n_fail) {
        printf("\n%d case(s) failed: an input flagged GGML_TENSOR_FLAG_INPUT was\n"
               "overwritten while a submitted graph could still read it.\n", n_fail);
        return 1;
    }
    printf("\nOK: every row written by the chunk that owns it\n");
    return 0;
}
