// CUDA kernels of the 'cuda' backend of desc_bpz.batched.
//
// They implement, for T = float or double, the same algorithm as
// _pipeline.py (see there and _prep.py for the maths):
//
//   bpz_chi2<T>       one thread per (galaxy, z, template): residual chi^2
//   bpz_posterior<T>  one block per galaxy: min chi^2, prior, log-space
//                     posterior, marginals, convolution, trimming,
//                     normalisation, modes, ODDS and confidence interval.
//
// Array layouts are C-contiguous: chi2/scratch (B, nz, nt), templates
// (nz, nt, nf), per-galaxy inputs (B, nf).

#define BLOCK 256

enum { PRIOR_NONE = 0, PRIOR_TABLE = 1, PRIOR_PARAMS = 2 };
enum { MODE_RAIL = 0, MODE_BPZ = 1 };
// Integer outputs per galaxy
enum { O_IZB, O_ITB, O_TBVALID, O_IZML, O_ITML, O_IZ1, O_IZ2, N_IOUT };
// Floating point outputs per galaxy
enum { O_CHI2, O_TODDS, O_ZMEAN, O_ODDS, N_FOUT };

template <typename T>
__global__ void bpz_chi2(const T* __restrict__ m, const T* __restrict__ y,
                         const T* __restrict__ s, const T* __restrict__ c,
                         int B, int nz, int nt, int nf, int expanded,
                         T* __restrict__ chi2) {
    const long long nzt = (long long)nz * nt;
    const long long n = (long long)B * nzt;
    for (long long e = blockIdx.x * (long long)blockDim.x + threadIdx.x; e < n;
         e += (long long)gridDim.x * blockDim.x) {
        const int g = (int)(e / nzt);
        const long long r = e - g * nzt;
        const T* mm = m + r * nf;
        const T* yy = y + (long long)g * nf;
        const T* ss = s + (long long)g * nf;
        T fot = 0, ftt = 0;
        for (int f = 0; f < nf; ++f) {
            const T x = mm[f] * ss[f];
            fot += x * yy[f];
            ftt += x * x;
        }
        T out;
        if (expanded) {
            T foo = 0;
            for (int f = 0; f < nf; ++f) foo += yy[f] * yy[f];
            foo += c[g];
            out = ftt > 0 ? foo - fot * fot / ftt : foo;
        } else {
            const T a = ftt > 0 ? fot / ftt : T(0);
            out = c[g];
            for (int f = 0; f < nf; ++f) {
                const T rr = yy[f] - a * (mm[f] * ss[f]);
                out += rr * rr;
            }
        }
        chi2[e] = out;
    }
}

// ---------------------------------------------------------------- block helpers

template <typename T> __device__ T neg_inf();
template <> __device__ float neg_inf<float>() { return __int_as_float(0xff800000); }
template <> __device__ double neg_inf<double>() { return __longlong_as_double(0xfff0000000000000ULL); }

template <typename T>
__device__ T block_sum(T v, T* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) {
        if (threadIdx.x < k) sh[threadIdx.x] += sh[threadIdx.x + k];
        __syncthreads();
    }
    const T r = sh[0];
    __syncthreads();
    return r;
}

template <typename T>
__device__ T block_max(T v, T* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) {
        if (threadIdx.x < k) sh[threadIdx.x] = max(sh[threadIdx.x], sh[threadIdx.x + k]);
        __syncthreads();
    }
    const T r = sh[0];
    __syncthreads();
    return r;
}

// Arg-extremum with numpy's tie rule (lowest index wins).  sign = +1 for
// argmax, -1 for argmin.  Invalid entries carry index INT_MAX.
template <typename T>
__device__ void block_argext(T& v, int& i, T sign, T* shv, int* shi) {
    shv[threadIdx.x] = v;
    shi[threadIdx.x] = i;
    __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) {
        if (threadIdx.x < k) {
            const T a = shv[threadIdx.x], b = shv[threadIdx.x + k];
            const int ia = shi[threadIdx.x], ib = shi[threadIdx.x + k];
            const bool take_b = (ib != 0x7fffffff) &&
                                (ia == 0x7fffffff || sign * b > sign * a || (b == a && ib < ia));
            if (take_b) {
                shv[threadIdx.x] = b;
                shi[threadIdx.x] = ib;
            }
        }
        __syncthreads();
    }
    v = shv[0];
    i = shi[0];
    __syncthreads();
}

template <typename T>
__device__ void thread_argext(T x, int ix, T& v, int& i, T sign) {
    if (i == 0x7fffffff || sign * x > sign * v || (x == v && ix < i)) {
        v = x;
        i = ix;
    }
}

// ---------------------------------------------------------------- posterior

template <typename T>
__global__ void bpz_posterior(
    T* __restrict__ work,  // (B, nz, nt): chi2 in, posterior out
    T* __restrict__ pbuf,  // (B, nz, nt): prior scratch (PRIOR_PARAMS only)
    int B, int nz, int nt,
    int prior_kind, const T* __restrict__ table, const int* __restrict__ index,
    const T* __restrict__ zt_at_a, const T* __restrict__ pa, const T* __restrict__ pzo,
    const T* __restrict__ pkm, const T* __restrict__ pkt, const T* __restrict__ pfo,
    int n_most, int n_last, const T* __restrict__ dm, T tail_thr,
    const T* __restrict__ z, const T* __restrict__ ker, int nk,
    int mode, T chi2_clip, T log_1e8, T log_1e300, T p_min, T oi_rms,
    T q1, T q2, int use_zthr, T z_thr, T zmax,
    T* __restrict__ pdf_out, int* __restrict__ iout, T* __restrict__ fout) {
    extern __shared__ unsigned char smem_raw[];
    T* pz = reinterpret_cast<T*>(smem_raw);  // nz
    T* pz2 = pz + nz;                         // nz
    T* tsum = pz2 + nz;                       // nt (prior norm, then tmarg)
    T* ft = tsum + nt;                        // nt (prior type fractions)
    __shared__ T shv[BLOCK];
    __shared__ int shi[BLOCK];
    __shared__ T s_scalar[2];

    const int g = blockIdx.x;
    if (g >= B) return;
    const int tid = threadIdx.x;
    const int nzt = nz * nt;
    T* w = work + (long long)g * nzt;

    // 1. min / argmin of chi^2 (first occurrence)
    T vmin = 0;
    int imin = 0x7fffffff;
    for (int e = tid; e < nzt; e += blockDim.x) thread_argext(w[e], e, vmin, imin, T(-1));
    block_argext(vmin, imin, T(-1), shv, shi);
    const T chi2_min = vmin;

    // 2. prior (prior_from_dict.prior_function, vectorised over templates)
    const T* P = nullptr;
    if (prior_kind == PRIOR_TABLE) {
        P = table + (long long)index[g] * nzt;
    } else if (prior_kind == PRIOR_PARAMS) {
        T* pb = pbuf + (long long)g * nzt;
        const T d = dm[g];
        for (int t = 0; t < nt; ++t) {
            const T zmt = pow(min(max(pzo[t] + pkm[t] * d, T(0.01)), T(15.0)), pa[t]);
            T acc = 0;
            for (int iz = tid; iz < nz; iz += blockDim.x) {
                const T zt = zt_at_a[iz * nt + t];
                const T p = zt * exp(T(-1.0) * min(max(zt / zmt, T(0)), T(700.0)));
                pb[iz * nt + t] = p;
                acc += p;
            }
            const T norm = block_sum(acc, shv);
            acc = 0;
            for (int iz = tid; iz < nz; iz += blockDim.x) {
                T q = pb[iz * nt + t] / norm;
                q = q < tail_thr ? T(0) : q;
                pb[iz * nt + t] = q;
                acc += q;
            }
            const T norm2 = block_sum(acc, shv);
            if (tid == 0) tsum[t] = norm2;
        }
        if (tid == 0) {
            T most = 0;
            for (int t = 0; t < nt; ++t) {
                ft[t] = (n_last == 0 || t < n_most) ? pfo[t] * exp(-pkt[t] * d) : T(0);
                if (t < n_most) most += ft[t];
            }
            if (n_last)
                for (int t = n_most; t < nt; ++t) ft[t] = (T(1.0) - most) / T(n_last);
        }
        __syncthreads();
        for (int e = tid; e < nzt; e += blockDim.x) {
            const int t = e % nt;
            pb[e] = pb[e] / tsum[t] * ft[t];
        }
        __syncthreads();
        P = pb;
    }

    // 3. log posterior and its per-galaxy maximum (the scale "shift")
    T lmax = neg_inf<T>();
    for (int e = tid; e < nzt; e += blockDim.x) {
        T lp = T(-0.5) * min(max(w[e] - chi2_min, T(0)), chi2_clip);
        if (P) lp += log(P[e]);
        w[e] = lp;
        lmax = max(lmax, lp);
    }
    T shift = block_max(lmax, shv);
    if (!isfinite(shift)) shift = 0;

    // 4. posterior, p(z) = sum_t post
    for (int iz = tid; iz < nz; iz += blockDim.x) {
        T acc = 0;
        for (int t = 0; t < nt; ++t) {
            const T p = exp(w[iz * nt + t] - shift);
            w[iz * nt + t] = p;
            acc += p;
        }
        pz[iz] = acc;
    }
    __syncthreads();
    // template marginals
    for (int t = 0; t < nt; ++t) {
        T acc = 0;
        for (int iz = tid; iz < nz; iz += blockDim.x) acc += w[iz * nt + t];
        const T s = block_sum(acc, shv);
        if (tid == 0) tsum[t] = s;
    }

    // 5. optional convolution, np.convolve(p, ker, 'same')
    T* p = pz;
    if (nk > 0) {
        const int right = (nk - 1) / 2;
        for (int iz = tid; iz < nz; iz += blockDim.x) {
            T acc = 0;
            for (int k = 0; k < nk; ++k) {
                const int j = iz + right - k;
                if (j >= 0 && j < nz) acc += ker[k] * pz[j];
            }
            pz2[iz] = acc;
        }
        __syncthreads();
        p = pz2;
    }

    // 6. mode, trimming, normalisation
    T vmax = 0;
    int izb = 0x7fffffff;
    for (int iz = tid; iz < nz; iz += blockDim.x) thread_argext(p[iz], iz, vmax, izb, T(1));
    block_argext(vmax, izb, T(1), shv, shi);
    const T thr = vmax * p_min;
    T acc = 0;
    for (int iz = tid; iz < nz; iz += blockDim.x) {
        T v = p[iz];
        v = (mode == MODE_BPZ) ? (v > thr ? v : T(0)) : (v < thr ? T(0) : v);
        p[iz] = v;
        acc += v;
    }
    const T tot = block_sum(acc, shv);
    acc = 0;
    for (int iz = tid; iz < nz; iz += blockDim.x) acc += z[iz] * p[iz];
    const T zsum = block_sum(acc, shv);

    T* out = pdf_out + (long long)g * nz;
    if (mode == MODE_BPZ) {
        for (int iz = tid; iz < nz; iz += blockDim.x) {
            p[iz] = p[iz] / tot;
            out[iz] = p[iz];
        }
    } else {
        const bool norm = tot > 0 && (log(tot) + shift > log_1e8);
        const T scale = exp(shift);
        for (int iz = tid; iz < nz; iz += blockDim.x) out[iz] = norm ? p[iz] / tot : p[iz] * scale;
    }
    __syncthreads();

    // 7. scalar statistics (serial, in numpy's summation order for cumsum)
    if (tid == 0) {
        const T* row = w + izb * nt;
        int itb = 0;
        T best = row[0];
        T tall = tsum[0];
        for (int t = 1; t < nt; ++t) {
            if (row[t] > best) {
                best = row[t];
                itb = t;
            }
            tall += tsum[t];
        }
        // the cumulative sum of p is only needed at a few indices
        const T zb = z[izb];
        T zo1, zo2;
        if (use_zthr) {
            zo1 = z_thr;
            zo2 = zmax;
        } else {
            zo1 = zb - oi_rms * (T(1.0) + zb);
            zo2 = zb + oi_rms * (T(1.0) + zb);
        }
        int i1 = -1, i2 = 0;
        for (int iz = 0; iz < nz; ++iz) {
            if (z[iz] < zo1) ++i1;
            if (z[iz] < zo2) ++i2;
        }
        T cp = 0, c_i1 = 0, c_i2 = 0;
        int j1 = 0, j2 = 0;
        for (int iz = 0; iz < nz; ++iz) {
            cp += p[iz];
            if (iz == i1) c_i1 = cp;
            if (iz == min(i2, nz - 1)) c_i2 = cp;
        }
        const T ctot = cp;
        cp = 0;
        for (int iz = 0; iz < nz; ++iz) {
            cp += p[iz];
            const T cpn = cp / ctot;
            if (cpn < q1) ++j1;
            if (cpn < q2) ++j2;
        }
        const T lower = i1 < 0 ? T(0) : c_i1;
        const T upper = i2 > nz - 1 ? ctot : c_i2;

        int* io = iout + (long long)g * N_IOUT;
        T* fo = fout + (long long)g * N_FOUT;
        io[O_IZB] = izb;
        io[O_ITB] = itb;
        io[O_TBVALID] = !(log(best) + shift < log_1e300);
        io[O_IZML] = imin / nt;
        io[O_ITML] = imin % nt;
        io[O_IZ1] = max(j1 - 1, 0);
        io[O_IZ2] = min(j2, nz - 1);
        fo[O_CHI2] = chi2_min;
        fo[O_TODDS] = tsum[itb] / tall;
        fo[O_ZMEAN] = zsum / tot;
        fo[O_ODDS] = (upper - lower) / ctot;
    }
}
