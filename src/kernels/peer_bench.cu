// src/kernels/peer_bench.cu - what a split hop costs, host-staged vs peer-to-peer, on this machine
//
//   build/peer_bench [--devices 0,1,2,3] [--all-pairs] [--sizes 64K,256K,128M] [--reps N] [--busy]
//
// A GLM layer split hands every token's residual (hc x n_embd floats: 64 KiB with the defaults) from one card to the
// next, today through a pinned host buffer: a D2H copy on the sender's stream, an event, the receiver's stream waiting
// on it, an H2D copy there.  This times that hop against the peer-to-peer ways of doing it, each the same shape (the
// sender's work, an event, the receiver waiting), wall clock from the first enqueue to the receiver's stream done:
//
//   event only   - no data: the floor every hop pays (the cross-device wait and the host's synchronize)
//   host staged  - D2H on the sender, H2D on the receiver (what maya does)
//   peer pull    - cudaMemcpyPeerAsync on the receiver's stream (the receiver's DMA reads the sender's VRAM)
//   peer push    - cudaMemcpyPeerAsync on the sender's stream (the sender's DMA writes the receiver's VRAM)
//   kernel pull  - a copy kernel on the receiver loading from the sender's VRAM
//   kernel push  - a copy kernel on the sender storing to the receiver's VRAM
//
// A pull row runs only where the receiver has peer access to the sender, a push row where the sender has it to the
// receiver (enabled first: a peer copy without it is staged through the host by the driver).  Each row's last pass
// is checked against the source (`check`).  The pairs are each card to the next (a split's hops) unless --all-pairs.
// --busy keeps a 64 MiB H2D copy running on the receiver all the while, as the expert uploads from RAM do during a
// decode - the DMA engines the copies share.
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {

bool ck(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::fprintf(stderr, "peer_bench: %s: %s\n", what, cudaGetErrorString(e));
    (void) cudaGetLastError();
    return false;
}

__global__ void copy_kernel(const uint4* __restrict__ src, uint4* __restrict__ dst, size_t n) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x)
        dst[i] = src[i];
}

__global__ void fill_kernel(uint32_t* p, size_t n, uint32_t seed) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x)
        p[i] = (uint32_t) i * 2654435761u ^ seed;
}

struct Dev {
    int id = 0, sms = 0;
    cudaStream_t s = nullptr;
    cudaEvent_t ev = nullptr;
};

enum Method { kEvent, kHost, kPeerPull, kPeerPush, kKernelPull, kKernelPush, kMethods };
const char* const kName[kMethods] = {"event only", "host staged", "peer pull", "peer push", "kernel pull",
                                     "kernel push"};

struct Bufs {
    void* src = nullptr;   // on the sender
    void* dst = nullptr;   // on the receiver
    void* host = nullptr;  // pinned
    size_t bytes = 0;
};

unsigned blocks_for(const Dev& d, size_t n16) {
    const size_t want = (n16 + 255) / 256;
    return (unsigned) std::max<size_t>(1, std::min<size_t>(want, (size_t) std::max(d.sms, 1) * 4));
}

// one hop A -> B by method m, enqueued (the caller synchronizes B's stream)
bool enqueue(Method m, Dev& A, Dev& B, Bufs& b) {
    const size_t n16 = b.bytes / sizeof(uint4);
    bool ok = true;
    cudaSetDevice(A.id);
    switch (m) {
        case kHost: ok = ck(cudaMemcpyAsync(b.host, b.src, b.bytes, cudaMemcpyDeviceToHost, A.s), "D2H"); break;
        case kPeerPush: ok = ck(cudaMemcpyPeerAsync(b.dst, B.id, b.src, A.id, b.bytes, A.s), "peer push"); break;
        case kKernelPush:
            copy_kernel<<<blocks_for(A, n16), 256, 0, A.s>>>((const uint4*) b.src, (uint4*) b.dst, n16);
            ok = ck(cudaGetLastError(), "kernel push");
            break;
        default: break;
    }
    ok = ok && ck(cudaEventRecord(A.ev, A.s), "event record");
    cudaSetDevice(B.id);
    ok = ok && ck(cudaStreamWaitEvent(B.s, A.ev, 0), "stream wait");
    switch (m) {
        case kHost: ok = ok && ck(cudaMemcpyAsync(b.dst, b.host, b.bytes, cudaMemcpyHostToDevice, B.s), "H2D"); break;
        case kPeerPull: ok = ok && ck(cudaMemcpyPeerAsync(b.dst, B.id, b.src, A.id, b.bytes, B.s), "peer pull"); break;
        case kKernelPull:
            copy_kernel<<<blocks_for(B, n16), 256, 0, B.s>>>((const uint4*) b.src, (uint4*) b.dst, n16);
            ok = ok && ck(cudaGetLastError(), "kernel pull");
            break;
        default: break;
    }
    return ok;
}

// the receiver's buffer equals the sender's (both copied back to the host)
bool check(Dev& A, Dev& B, Bufs& b) {
    std::vector<uint32_t> s(b.bytes / 4), d(b.bytes / 4);
    cudaSetDevice(A.id);
    bool ok = ck(cudaMemcpy(s.data(), b.src, b.bytes, cudaMemcpyDeviceToHost), "check src");
    cudaSetDevice(B.id);
    ok = ok && ck(cudaMemcpy(d.data(), b.dst, b.bytes, cudaMemcpyDeviceToHost), "check dst");
    return ok && std::memcmp(s.data(), d.data(), b.bytes) == 0;
}

std::string size_str(size_t b) {
    char buf[32];
    if (b >= ((size_t) 1 << 20) && b % ((size_t) 1 << 20) == 0) std::snprintf(buf, sizeof buf, "%zu MiB", b >> 20);
    else if (b >= 1024 && b % 1024 == 0) std::snprintf(buf, sizeof buf, "%zu KiB", b >> 10);
    else std::snprintf(buf, sizeof buf, "%zu B", b);
    return buf;
}

size_t parse_size(const std::string& t) {
    char* end = nullptr;
    double v = std::strtod(t.c_str(), &end);
    switch (end && *end ? *end : ' ') {
        case 'k': case 'K': v *= 1024; break;
        case 'm': case 'M': v *= 1024 * 1024; break;
        case 'g': case 'G': v *= 1024.0 * 1024 * 1024; break;
        default: break;
    }
    return ((size_t) v + 15) & ~(size_t) 15;
}

std::vector<std::string> split_list(const char* s) {
    std::vector<std::string> out;
    std::string cur;
    for (const char* p = s; ; ++p) {
        if (*p == ',' || *p == '\0') {
            if (!cur.empty()) out.push_back(cur);
            cur.clear();
            if (*p == '\0') break;
        } else {
            cur += *p;
        }
    }
    return out;
}

// --busy: a 64 MiB H2D copy after another on the receiver, on a stream of its own, until stop
struct Busy {
    std::atomic<bool> stop{false};
    std::thread th;
    void start(int dev) {
        stop = false;
        th = std::thread([this, dev] {
            const size_t bytes = (size_t) 64 << 20;
            void *h = nullptr, *d = nullptr;
            cudaStream_t s = nullptr;
            cudaSetDevice(dev);
            if (ck(cudaHostAlloc(&h, bytes, cudaHostAllocPortable), "busy host") &&
                ck(cudaMalloc(&d, bytes), "busy dev") &&
                ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "busy stream"))
                while (!stop.load()) {
                    if (!ck(cudaMemcpyAsync(d, h, bytes, cudaMemcpyHostToDevice, s), "busy copy")) break;
                    cudaStreamSynchronize(s);
                }
            if (s) cudaStreamDestroy(s);
            if (d) cudaFree(d);
            if (h) cudaFreeHost(h);
        });
    }
    void end() {
        stop = true;
        if (th.joinable()) th.join();
    }
};

}  // namespace

int main(int argc, char** argv) {
    std::vector<int> devs;
    std::vector<size_t> sizes = {(size_t) 64 << 10, (size_t) 256 << 10, (size_t) 128 << 20};
    int reps = 0;   // 0: by size (200 small, 20 large)
    bool all_pairs = false, busy = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--devices" && i + 1 < argc) {
            for (auto& t : split_list(argv[++i])) devs.push_back(std::atoi(t.c_str()));
        } else if (a == "--sizes" && i + 1 < argc) {
            sizes.clear();
            for (auto& t : split_list(argv[++i])) sizes.push_back(parse_size(t));
        } else if (a == "--reps" && i + 1 < argc) {
            reps = std::max(1, std::atoi(argv[++i]));
        } else if (a == "--all-pairs") {
            all_pairs = true;
        } else if (a == "--busy") {
            busy = true;
        } else {
            std::fprintf(stderr,
                         "usage: %s [--devices 0,1,...] [--all-pairs] [--sizes 64K,256K,128M] [--reps N] [--busy]\n",
                         argv[0]);
            return 2;
        }
    }
    int n_dev = 0;
    if (!ck(cudaGetDeviceCount(&n_dev), "device count")) return 1;
    if (devs.empty())
        for (int d = 0; d < n_dev; ++d) devs.push_back(d);
    for (int d : devs)
        if (d < 0 || d >= n_dev) {
            std::fprintf(stderr, "peer_bench: no device %d (%d here)\n", d, n_dev);
            return 2;
        }
    if (devs.size() < 2) {
        std::printf("peer_bench: needs two GPUs (%d here)\n", n_dev);
        return 0;
    }

    std::vector<Dev> D((size_t) n_dev);
    for (int d : devs) {
        Dev& x = D[(size_t) d];
        x.id = d;
        cudaSetDevice(d);
        cudaDeviceProp p{};
        cudaGetDeviceProperties(&p, d);
        char bus[32] = "?";
        cudaDeviceGetPCIBusId(bus, sizeof bus, d);
        cudaDeviceGetAttribute(&x.sms, cudaDevAttrMultiProcessorCount, d);
        if (!ck(cudaStreamCreateWithFlags(&x.s, cudaStreamNonBlocking), "stream") ||
            !ck(cudaEventCreateWithFlags(&x.ev, cudaEventDisableTiming), "event"))
            return 1;
#if defined(__HIP_PLATFORM_AMD__)
        std::printf("GPU%d: %s (%s), %s, large BAR %s\n", d, p.name, p.gcnArchName, bus, p.isLargeBar ? "yes" : "no");
#else
        std::printf("GPU%d: %s, %s\n", d, p.name, bus);
#endif
    }
#if defined(__HIP_PLATFORM_AMD__)
    const char* sdma = std::getenv("HSA_ENABLE_SDMA");
    std::printf("HSA_ENABLE_SDMA=%s\n", sdma ? sdma : "(unset)");
#endif
    // peer access both ways wherever the driver offers it
    std::vector<std::vector<int>> peer((size_t) n_dev, std::vector<int>((size_t) n_dev, 0));
    for (int a : devs)
        for (int b : devs) {
            if (a == b) continue;
            int can = 0;
            if (cudaDeviceCanAccessPeer(&can, a, b) != cudaSuccess) can = 0;   // a can map b's memory
            if (can) {
                cudaSetDevice(a);
                const cudaError_t e = cudaDeviceEnablePeerAccess(b, 0);
                can = e == cudaSuccess || e == cudaErrorPeerAccessAlreadyEnabled;
            }
            (void) cudaGetLastError();
            peer[(size_t) a][(size_t) b] = can;
        }
    std::vector<std::pair<int, int>> pairs;
    for (size_t i = 0; i < devs.size(); ++i)
        for (size_t j = 0; j < devs.size(); ++j)
            if (i != j && (all_pairs || j == i + 1)) pairs.emplace_back(devs[i], devs[j]);
    if (busy) std::printf("--busy: a 64 MiB H2D copy runs on the receiver throughout\n");
    std::printf("wall time per hop, from the first enqueue to the receiver's stream done (host sync included)\n");

    int failures = 0;
    for (auto [a, b] : pairs) {
        Dev& A = D[(size_t) a];
        Dev& B = D[(size_t) b];
        // a pull row runs where the receiver maps the sender's memory, a push row where the sender maps the
        // receiver's (a peer copy without access enabled is staged through the host by the driver: not P2P)
        const bool pull_ok = peer[(size_t) b][(size_t) a] != 0, push_ok = peer[(size_t) a][(size_t) b] != 0;
        std::printf("\nGPU%d -> GPU%d  (peer access: GPU%d maps GPU%d %s, GPU%d maps GPU%d %s)\n", a, b, b, a,
                    pull_ok ? "yes" : "no", a, b, push_ok ? "yes" : "no");
        std::printf("  %-8s  %-12s %10s %10s %9s  %s\n", "size", "method", "median us", "p90 us", "GB/s", "check");
        Busy bz;
        if (busy) bz.start(b);
        for (size_t bytes : sizes) {
            Bufs bf;
            bf.bytes = bytes;
            cudaSetDevice(a);
            bool ok = ck(cudaMalloc(&bf.src, bytes), "src") &&
                      ck(cudaHostAlloc(&bf.host, bytes, cudaHostAllocPortable), "host");
            cudaSetDevice(b);
            ok = ok && ck(cudaMalloc(&bf.dst, bytes), "dst");
            if (!ok) {
                ++failures;
                continue;
            }
            const int n = reps > 0 ? reps : (bytes >= ((size_t) 16 << 20) ? 20 : 200);
            for (int m = 0; m < kMethods; ++m) {
                const Method mm = (Method) m;
                const bool pull = mm == kPeerPull || mm == kKernelPull, push = mm == kPeerPush || mm == kKernelPush;
                if ((pull && !pull_ok) || (push && !push_ok)) continue;
                // a fresh pattern on the sender, the receiver cleared: the check sees this row's copies only
                cudaSetDevice(a);
                fill_kernel<<<256, 256, 0, A.s>>>((uint32_t*) bf.src, bytes / 4, 0x5eed0000u + (uint32_t) m);
                ck(cudaStreamSynchronize(A.s), "fill");
                cudaSetDevice(b);
                ck(cudaMemset(bf.dst, 0, bytes), "clear");
                ck(cudaDeviceSynchronize(), "clear sync");
                std::vector<double> us;
                us.reserve((size_t) n);
                bool run_ok = true;
                for (int r = -std::max(5, n / 10); r < n && run_ok; ++r) {   // (the first passes untimed)
                    const auto t0 = std::chrono::steady_clock::now();
                    run_ok = enqueue(mm, A, B, bf);
                    cudaSetDevice(b);
                    run_ok = run_ok && ck(cudaStreamSynchronize(B.s), "sync");
                    const auto t1 = std::chrono::steady_clock::now();
                    const double t = std::chrono::duration<double, std::micro>(t1 - t0).count();
                    if (r >= 0) us.push_back(t);
                }
                if (!run_ok || us.empty()) {
                    std::printf("  %-8s  %-12s %10s\n", size_str(bytes).c_str(), kName[m], "failed");
                    ++failures;
                    continue;
                }
                std::sort(us.begin(), us.end());
                const double med = us[us.size() / 2], p90 = us[std::min(us.size() - 1, us.size() * 9 / 10)];
                const char* chk = "-";
                if (mm != kEvent) {
                    const bool good = check(A, B, bf);
                    chk = good ? "ok" : "MISMATCH";
                    if (!good) ++failures;
                }
                char gbs[16] = "-";
                if (mm != kEvent) std::snprintf(gbs, sizeof gbs, "%.1f", (double) bytes / (med * 1e3));
                std::printf("  %-8s  %-12s %10.1f %10.1f %9s  %s\n", size_str(bytes).c_str(), kName[m], med, p90, gbs,
                            chk);
                std::fflush(stdout);
            }
            cudaSetDevice(a);
            cudaFree(bf.src);
            cudaFreeHost(bf.host);
            cudaSetDevice(b);
            cudaFree(bf.dst);
        }
        bz.end();
    }
    return failures == 0 ? 0 : 1;
}
