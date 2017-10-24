import numpy as np

def sigmoid(x):
  """
  A numerically stable version of the logistic sigmoid function.
  """
  pos_mask = (x >= 0)
  neg_mask = (x < 0)
  z = np.zeros_like(x)
  z[pos_mask] = np.exp(-x[pos_mask])
  z[neg_mask] = np.exp(x[neg_mask])
  top = np.ones_like(x)
  top[neg_mask] = z[neg_mask]
  return top / (1 + z)


def sru_step_forward(x, prev_c, W, Wf, Wr, Wh, bf, br):
    """
    Inputs:
    - x: (N, D)
    """
    ax = x.dot(W)
    af = x.dot(Wf) + bf
    ar = x.dot(Wr) + br
    ah = x.dot(Wh)
    f = sigmoid(af)
    r = sigmoid(ar)
    next_c = f * prev_c + (1 - f) * ax
    gc = next_c
    h = r * gc + (1 - r) * ah

    cache = x, prev_c, W, Wf, Wr, Wh, bf, br, ax, af, ar, ah, f, r, gc
    return h, next_c, cache

def sru_step_backward(dh, dnext_c, cache):
    x, pc, W, Wf, Wr, Wh, bf, br, ax, af, ar, ah, f, r, gc = cache

    dr = dh * (gc - ah)
    dgc = dh * r
    dah = dh * (1 - r)
    dnc = dgc + dnext_c  # NOTE HERE

    df = dnc * (pc - ax)
    dpc = dnc * f
    dax = dnc * (1 - f)
    dar = dr * (1 - r) * r
    daf = df * (1 - f) * f

    dWh = np.dot(x.T, dah)
    dWr = np.dot(x.T, dar)
    dWf = np.dot(x.T, daf)
    dW = np.dot(x.T, dax)
    dbf = daf.sum(axis=0)
    dbr = dar.sum(axis=0)

    dx = dah.dot(Wh.T) + dar.dot(Wr.T) + daf.dot(Wf.T) + dax.dot(W.T)
    return dx, dpc, dW, dWf, dWr, dWh, dbf, dbr

def sru_forward(x, W, Wf, Wr, Wh, bf, br):
    """
    Inputs:
    - x: (N, T, D)
    - W: (D, H)
    - Wf: (D, H)
    - Wr: (D, H)
    - Wh: (D, H)

    Returns a tuple of:
    - h: (N, T, H)
    """
    N, T, D = x.shape
    _, H = W.shape

    h = np.zeros((N, T, H))
    cache = []

    prev_c = np.zeros((N, H))
    for t in range(T):
        h[:, t, :], next_c, cache_t = sru_step_forward(x[:, t, :], prev_c,
                                                       W, Wf, Wr, Wh, bf, br)
        prev_c = next_c
        cache.append(cache_t)
    return h, cache

def sru_backward(dh, cache):
    """
    Inputs:
    - dh: (N, T, H)
    """
    N, T, H = dh.shape
    N, D = cache[0][0].shape

    dx = np.zeros((N, T, D))
    dc0 = np.zeros((N, H))
    dW = np.zeros((D, H))
    dWf = np.zeros((D, H))
    dWr = np.zeros((D, H))
    dWh = np.zeros((D, H))
    dbf = np.zeros((H))
    dbr = np.zeros((H))

    dnext_c = np.zeros((N, H))
    for t in xrange(T-1, -1, -1): # t = T-1, T-2, ..., 0
        dxt, dpct, dWt, dWft, dWrt, dWht, dbft, dbrt = sru_step_backward(dh[:, t, :],
                                                                         dnext_c,
                                                                         cache[t])
        dnext_c = dpct
        dx[:, t, :] = dxt
        dW += dWt
        dWf += dWft
        dWr += dWrt
        dWh += dWht
        dbf += dbft
        dbr += dbrt
    # dc0 = dpct
    return dx, dW, dWf, dWr, dWh, dbf, dbr


def sru_step_forward_fast(x, prev_c, W, bf, br):
    """
    Inputs:
    - x: (N, D)
    - W: (D, 4*H)
    """
    a = x.dot(W)
    ax, af, ar, ah = np.array_split(a, 4, axis=1)
    af = af + bf
    ar = ar + br
    f = sigmoid(af)
    r = sigmoid(ar)
    next_c = f * prev_c + (1 - f) * ax
    gc = next_c
    h = r * gc + (1 - r) * ah

    cache = x, prev_c, W, bf, br, ax, ah, f, r, gc
    return h, next_c, cache

def sru_step_backward_fast(dh, dnext_c, cache):
    x, pc, W, bf, br, ax, ah, f, r, gc = cache

    dr = dh * (gc - ah)
    dgc = dh * r
    dah = dh * (1 - r)
    dnc = dgc + dnext_c  # NOTE HERE

    df = dnc * (pc - ax)
    dpc = dnc * f
    dax = dnc * (1 - f)
    dar = dr * (1 - r) * r
    daf = df * (1 - f) * f

    da = np.hstack((dax, daf, dar, dah))
    dW = np.dot(x.T, da)
    dbf = daf.sum(axis=0)
    dbr = dar.sum(axis=0)

    dx = da.dot(W.T)
    return dx, dpc, dW, dbf, dbr


def sru_forward_fast(x, W, bf, br):
    """
    Inputs:
    - x: (N, T, D)
    - W: (D, 4*H)
    """
    N, T, D = x.shape
    H = W.shape[1] // 4
    h = np.zeros((T, N, H))  # at the end, np.transpose(h, (1, 0, 2))
    cache = []

    xt = np.transpose(x, (1, 0, 2))  # T, N, D
    x2d = np.reshape(xt, (-1, x.shape[-1]))  # NTxD
    a = x2d.dot(W)  # NTx4H
    prev_c = np.zeros((N, H))
    for t in range(T):  # 0, 1, ... , T-1
        ######
        # every step INPUT: at, prev_c
        ######
        at = a[t*N:(t+1)*N, :]  # Nx4H
        ax, af, ar, ah = np.array_split(at, 4, axis=1)
        af = af + bf
        ar = ar + br
        f = sigmoid(af)
        r = sigmoid(ar)
        next_c = f * prev_c + (1 - f) * ax
        gc = next_c
        h[t] = r * gc + (1 - r) * ah  # NxH
        cache.append((prev_c, ax, ah, f, r, gc))
        ######
        # every step OUTPUT: h[t], next_c, cache[t]
        ######
        # remember state c
        prev_c = next_c
    h = np.transpose(h, (1, 0, 2))  # N, T, H
    cache.append((x2d, a, W, bf, br))  # the last element
    return h, cache

def sru_backward_fast(dh, cache):
    """
    Inputs:
    - dh: (N, T, H)
    """
    N, T, H = dh.shape
    D = cache[-1][0].shape[-1]
    dc0 = np.zeros((N, H))
    dW = np.zeros((D, 4*H))
    dbf = np.zeros((H))
    dbr = np.zeros((H))

    da = np.zeros((T*N, 4*H))

    dht = np.transpose(dh, (1, 0, 2))  # T, N, H
    dh2d = np.reshape(dht, (-1, dh.shape[-1]))  # NTxH
    dnext_c = np.zeros((N, H))
    for t in xrange(T-1, -1, -1): # t = T-1, T-2, ..., 0
        pc, ax, ah, f, r, gc = cache[t]
        dh = dh2d[t*N:(t+1)*N, :]  # NxH

        dr = dh * (gc - ah)
        dgc = dh * r
        dah = dh * (1 - r)
        dnc = dgc + dnext_c

        df = dnc * (pc - ax)
        dpc = dnc * f
        dax = dnc * (1 - f)
        dar = dr * (1 - r) * r
        daf = df * (1 - f) * f
        
        da[t*N:(t+1)*N] = np.hstack((dax, daf, dar, dah))

        dnext_c = dpc
    x2d, a, W, bf, br = cache[-1]
    dx = da.dot(W.T)  # NTxD
    dW = np.dot(x2d.T, da)
    dbf = da[:, H:2*H].sum(axis=0)
    dbr = da[:, 2*H:3*H].sum(axis=0)

    dx = np.reshape(dx, (T, N, D))
    dx = np.transpose(dx, (1, 0, 2))  # N, T, H
    return dx, dW, dbf, dbr


def sru_backward_fast_advance(dh, cache):
    """
    Inputs:
    - dh: (N, T, H)
    """
    N, T, H = dh.shape
    D = cache[-1][0].shape[-1]
    dc0 = np.zeros((N, H))
    dW = np.zeros((D, 4*H))
    dbf = np.zeros((H))
    dbr = np.zeros((H))

    da = np.zeros((T*N, 4*H))

    dht = np.transpose(dh, (1, 0, 2))  # T, N, H
    dh2d = np.reshape(dht, (-1, dh.shape[-1]))  # NTxH
    dnc = np.zeros((N, H))
    next_f = np.zeros((N, H))
    for t in xrange(T-1, -1, -1):  # t = T-1, T-2, ..., 0
        pc, ax, ah, f, r, gc = cache[t]
        dh = dh2d[t*N:(t+1)*N, :]  # NxH

        dr = dh * (gc - ah)
        dgc = dh * r
        dah = dh * (1 - r)
        dnc = dnc * next_f
        dnc += dgc
        # advance dnc

        df = dnc * (pc - ax)
        # dpc = dnc * f
        dax = dnc * (1 - f)
        dar = dr * (1 - r) * r
        daf = df * (1 - f) * f
        
        da[t*N:(t+1)*N] = np.hstack((dax, daf, dar, dah))
        next_f = f
    x2d, a, W, bf, br = cache[-1]
    dx = da.dot(W.T)  # NTxD
    dW = np.dot(x2d.T, da)
    dbf = da[:, H:2*H].sum(axis=0)
    dbr = da[:, 2*H:3*H].sum(axis=0)

    dx = np.reshape(dx, (T, N, D))
    dx = np.transpose(dx, (1, 0, 2))  # N, T, H
    return dx, dW, dbf, dbr