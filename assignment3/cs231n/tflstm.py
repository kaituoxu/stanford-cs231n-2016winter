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


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
  """
  Forward pass for a single timestep of an LSTM.
  
  The input data has dimension D, the hidden state has dimension H, and we use
  a minibatch size of N.
  
  Inputs:
  - x: Input data, of shape (N, D)
  - prev_h: Previous hidden state, of shape (N, H)
  - prev_c: previous cell state, of shape (N, H)
  - Wx: Input-to-hidden weights, of shape (D, 4H)
  - Wh: Hidden-to-hidden weights, of shape (H, 4H)
  - b: Biases, of shape (4H,)
  
  Returns a tuple of:
  - next_h: Next hidden state, of shape (N, H)
  - next_c: Next cell state, of shape (N, H)
  - cache: Tuple of values needed for backward pass.
  """
  next_h, next_c, cache = None, None, None
  #############################################################################
  # TODO: Implement the forward pass for a single timestep of an LSTM.        #
  # You may want to use the numerically stable sigmoid implementation above.  #
  #############################################################################
  a = np.dot(x, Wx) + np.dot(prev_h, Wh) + b
  ai, af, ao, ag = np.array_split(a, 4, axis=1)
  i = sigmoid(ai)
  f = sigmoid(af)
  o = sigmoid(ao)
  g = np.tanh(ag)
  next_c = f * prev_c + i * g
  tnext_c = np.tanh(next_c)
  next_h = o * tnext_c
  
  cache = x, prev_h, prev_c, Wx, Wh, i, f, o, g, tnext_c
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  
  return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
  """
  Backward pass for a single timestep of an LSTM.
  
  Inputs:
  - dnext_h: Gradients of next hidden state, of shape (N, H)
  - dnext_c: Gradients of next cell state, of shape (N, H)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient of input data, of shape (N, D)
  - dprev_h: Gradient of previous hidden state, of shape (N, H)
  - dprev_c: Gradient of previous cell state, of shape (N, H)
  - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
  - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
  - db: Gradient of biases, of shape (4H,)
  """
  dx, dh, dc, dWx, dWh, db = None, None, None, None, None, None
  #############################################################################
  # TODO: Implement the backward pass for a single timestep of an LSTM.       #
  #                                                                           #
  # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
  # the output value from the nonlinearity.                                   #
  #############################################################################
  x, prev_h, prev_c, Wx, Wh, i, f, o, g, tnext_c = cache
  dnext_c += dnext_h * o * (1-tnext_c**2)

  dg = dnext_c * i
  do = dnext_h * tnext_c
  df = dnext_c * prev_c
  di = dnext_c * g

  dai = di * i * (1-i)
  daf = df * f * (1-f)
  dao = do * o * (1-o)
  dag = dg * (1-g**2)
  da = np.hstack((dai, daf, dao, dag))

  dx = np.dot(da, Wx.T)
  dprev_h = np.dot(da, Wh.T)
  dprev_c = dnext_c * f
  dWx = np.dot(x.T, da)
  dWh = np.dot(prev_h.T, da)
  db = da.sum(axis=0)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
  """
  Forward pass for an LSTM over an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the LSTM forward, we return the hidden states for all timesteps.
  
  Note that the initial cell(hidden?) state is passed as input, but the initial cell
  state is set to zero. Also note that the cell state is not returned; it is
  an internal variable to the LSTM and is not accessed from outside.
  
  Inputs:
  - x: Input data of shape (N, T, D)
  - h0: Initial hidden state of shape (N, H)
  - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
  - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
  - b: Biases of shape (4H,)
  
  Returns a tuple of:
  - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
  - cache: Values needed for the backward pass.
  """
  h, cache = None, None
  #############################################################################
  # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
  # You should use the lstm_step_forward function that you just defined.      #
  #############################################################################
  N, T, D = x.shape
  _, H = h0.shape

  h = np.zeros((N, T, H))
  cache = []

  prev_h = h0
  prev_c = np.zeros((N, H))
  for t in xrange(T):
    xt = x[:,t,:]
    a = np.dot(xt, Wx) + np.dot(prev_h, Wh) + b
    ai, af, ao, ag = np.array_split(a, 4, axis=1)
    i = sigmoid(ai)
    f = sigmoid(af)
    o = sigmoid(ao)
    g = np.tanh(ag)
    next_c = f * prev_c + i * g
    tnext_c = np.tanh(next_c)
    next_h = o * tnext_c
    cc_t = (xt, prev_h, prev_c, Wx, Wh, i, f, o, g, tnext_c)
    h[:,t,:], nc = next_h, next_c
    prev_h, prev_c = h[:,t,:], nc
    cache.append(cc_t)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return h, cache


def lstm_backward_advance_origin(dh, cache):
  dx, dh0, dWx, dWh, db = None, None, None, None, None
  N, T, H = dh.shape
  _, D = cache[0][0].shape
  dx = np.zeros((N, T, D))
  dh0 = np.zeros((N, H))
  dWx = np.zeros((D, 4*H))
  dWh = np.zeros((H, 4*H))
  db = np.zeros(4*H)

  dnext_h = np.zeros((N, H))
  dnext_c = np.zeros((N, H))

  for t in xrange(T-1, -1, -1): # t = T-1, T-2, ..., 0
    x, prev_h, prev_c, Wx, Wh, i, f, o, g, tnext_c = cache[t]
    dnext_h = dnext_h + dh[:, t, :]
    dnext_c += dnext_h * o * (1-tnext_c**2)

    dg = dnext_c * i
    do = dnext_h * tnext_c
    df = dnext_c * prev_c
    di = dnext_c * g

    dai = di * i * (1-i)
    daf = df * f * (1-f)
    dao = do * o * (1-o)
    dag = dg * (1-g**2)
    da = np.hstack((dai, daf, dao, dag))

    dxt = np.dot(da, Wx.T)
    dprev_h = np.dot(da, Wh.T)
    dprev_c = dnext_c * f
    dWxt = np.dot(x.T, da)
    dWht = np.dot(prev_h.T, da)
    dbt = da.sum(axis=0)
    dx[:,t,:] = dxt
    dnext_h, dnext_c = dprev_h, dprev_c
    dWx += dWxt
    dWh += dWht
    db += dbt
  dh0 = dnext_h
  return dx, dh0, dWx, dWh, db

def lstm_backward_advance(dh, cache):
  dx, dh0, dWx, dWh, db = None, None, None, None, None
  N, T, H = dh.shape
  _, D = cache[0][0].shape
  dx = np.zeros((N, T, D))
  dh0 = np.zeros((N, H))
  dWx = np.zeros((D, 4*H))
  dWh = np.zeros((H, 4*H))
  db = np.zeros(4*H)

  dnext_h = np.zeros((N, H))
  dnext_c = np.zeros((N, H))
  dnext_a = np.zeros((N, 4*H))
  next_f = np.zeros((N, H))

  for t in xrange(T-1, -1, -1): # t = T-1, T-2, ..., 0
    x, prev_h, prev_c, Wx, Wh, i, f, o, g, tnext_c = cache[t]
    dnext_h = dh[:, t, :]
    dnext_h = dnext_h + np.dot(dnext_a, Wh.T)

    dht = dnext_h * o * (1-tnext_c**2)

    do = dnext_h * tnext_c
    dao = do * o * (1-o)

    dct = dht
    dct = dct + dnext_c * next_f

    dg = dct * i
    df = dct * prev_c
    di = dct * g

    dai = di * i * (1-i)
    daf = df * f * (1-f)
    dag = dg * (1-g**2)
    da = np.hstack((dai, daf, dao, dag))

    dnext_c = dct
    dnext_a = da
    next_f = f

    dxt = np.dot(da, Wx.T)
    dWxt = np.dot(x.T, da)
    dWht = np.dot(prev_h.T, da)
    dbt = da.sum(axis=0)
    dx[:,t,:] = dxt
    dWx += dWxt
    dWh += dWht
    db += dbt
  dh0 = dnext_h
  return dx, dh0, dWx, dWh, db




def lstm_backward(dh, cache):
  """
  Backward pass for an LSTM over an entire sequence of data.]
  
  Inputs:
  - dh: Upstream gradients of hidden states, of shape (N, T, H)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient of input data of shape (N, T, D)
  - dh0: Gradient of initial hidden state of shape (N, H)
  - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
  - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
  - db: Gradient of biases, of shape (4H,)
  """
  dx, dh0, dWx, dWh, db = None, None, None, None, None
  #############################################################################
  # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
  # You should use the lstm_step_backward function that you just defined.     #
  #############################################################################
  N, T, H = dh.shape
  _, D = cache[0][0].shape
  dx = np.zeros((N, T, D))
  dh0 = np.zeros((N, H))
  dWx = np.zeros((D, 4*H))
  dWh = np.zeros((H, 4*H))
  db = np.zeros(4*H)


  dnext_h = np.zeros((N, H))
  dnext_c = np.zeros((N, H))

  for t in xrange(T-1, -1, -1): # t = T-1, T-2, ..., 0
    dxt, dprev_h, dprev_c, dWxt, dWht, dbt = \
        lstm_step_backward(dnext_h + dh[:, t, :], dnext_c, cache[t])
    dnext_h, dnext_c = dprev_h, dprev_c
    dx[:,t,:] = dxt
    dWx += dWxt
    dWh += dWht
    db += dbt
  dh0 = dnext_h
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  
  return dx, dh0, dWx, dWh, db


def tflstm_forward(x, h0, Wx, Wh, Wk, b, F, S):
  """
  Forward pass for an LSTM over an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the LSTM forward, we return the hidden states for all timesteps.
  
  Note that the initial cell(hidden?) state is passed as input, but the initial cell
  state is set to zero. Also note that the cell state is not returned; it is
  an internal variable to the LSTM and is not accessed from outside.
  
  Inputs:
  - x: Input data of shape (N, T, D)
  - h0: Initial hidden state of shape (N, H)
  - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
  - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
  - b: Biases of shape (4H,)
  
  Returns a tuple of:
  - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
  - cache: Values needed for the backward pass.
  """
  h, cache = None, None
  #############################################################################
  # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
  # You should use the lstm_step_forward function that you just defined.      #
  #############################################################################
  N, T, D = x.shape

  cache = []

  B = (D-F)/S+1
  _, H = h0.shape
  H = H / B
  h = np.zeros((N, T, B*H))
  prev_h = h0
  prev_c = np.zeros((N, B*H))
  for t in xrange(T):
    prev_k = np.zeros((N, H))
    cc_t = []
    for b in xrange(B):
      # Get inputs
      xtk = x[:,t,b*S:b*S+F] # NxF
      bdh = prev_h[:,b*H:(b+1)*H] # NxH, b = block
      bdc = prev_c[:,b*H:(b+1)*H] # NxH

      a = np.dot(xtk, Wx) + np.dot(bdh, Wh) + np.dot(prev_k, Wk) + b
      ai, af, ao, ag = np.array_split(a, 4, axis=1)
      i = sigmoid(ai)
      f = sigmoid(af)
      o = sigmoid(ao)
      g = np.tanh(ag)
      ctk = f * bdc + i * g
      tc = np.tanh(ctk)
      htk = o * tc
      cc_tk = (xtk, bdh, bdc, prev_k, Wx, Wh, Wk, i, f, o, g, tc)
      h[:,t,b*H:(b+1)*H] = htk
      prev_h[:,b*H:(b+1)*H] = htk
      prev_c[:,b*H:(b+1)*H] = ctk
      prev_k = htk
      cc_t.append(cc_tk)
    cache.append(cc_t)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  cache.append((D, F, S))
  return h, cache


def tflstm_backward_advance(dh, cache):
  dx, dh0, dWx, dWh, dWk, db = None, None, None, None, None, None
  N, T, H = dh.shape  # H = H / B
  D, F, S = cache[-1]
  B = (D-F)/S+1
  H = H / B

  dx = np.zeros((N, T, D))
  dh0 = np.zeros((N, B*H))
  dWx = np.zeros((F, 4*H))
  dWh = np.zeros((H, 4*H))
  dWk = np.zeros((H, 4*H))
  db = np.zeros(4*H)

  dnext_ct = np.zeros((N, B*H))
  dnext_at = np.zeros((N, B*4*H))
  next_f = np.zeros((N, B*H))

  for t in xrange(T-1, -1, -1): # t = T-1, T-2, ..., 0
    cc_t = cache[t]
    dnext_k = np.zeros((N, H))
    dnext_ak = np.zeros((N, 4*H))
    for b in xrange(B-1, -1, -1): # b = B-1, B-2, ..., 0
      xtk, bdh, bdc, prev_k, Wx, Wh, Wk, i, f, o, g, tc = cc_t[b]

      dnext_htk = dh[:, t, b*H:(b+1)*H]
      dnext_htk = dnext_htk + np.dot(dnext_at[:, b*4*H:(b+1)*4*H], Wh.T) + np.dot(dnext_ak, Wk.T)

      dhtk = dnext_htk * o * (1-tc**2)  # dctk

      do = dnext_htk * tc
      dao = do * o * (1-o)

      dctk = dhtk
      dctk = dctk + dnext_ct[:,b*H:(b+1)*H] * next_f[:,b*H:(b+1)*H]

      dg = dctk * i
      df = dctk * bdc
      di = dctk * g

      dai = di * i * (1-i)
      daf = df * f * (1-f)
      dag = dg * (1-g**2)
      da = np.hstack((dai, daf, dao, dag))

      dnext_ct[:, b*H:(b+1)*H] = dctk
      dnext_at[:, b*4*H:(b+1)*4*H] = da
      next_f[:, b*H:(b+1)*H] = f
      dnext_ak = da

      dxtk = np.dot(da, Wx.T) # NxF
      dWxtk = np.dot(xtk.T, da) # FxN dot Nx4H = Fx4H
      dWhtk = np.dot(bdh.T, da)
      dWktk = np.dot(prev_k.T, da)
      dbtk = da.sum(axis=0)
      dx[:,t,b*S:b*S+F] += dxtk
      dWx += dWxtk
      dWh += dWhtk
      dWk += dWktk
      db += dbtk
   #dh0 = dnext_h
  return dx, dh0, dWx, dWh, dWk, db


def tflstm_forward_origin(x, h0, Wx, Wh, Wk, b, F, S):
  """
  Forward pass for an LSTM over an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the LSTM forward, we return the hidden states for all timesteps.
  
  Note that the initial cell(hidden?) state is passed as input, but the initial cell
  state is set to zero. Also note that the cell state is not returned; it is
  an internal variable to the LSTM and is not accessed from outside.
  
  Inputs:
  - x: Input data of shape (N, T, D)
  - h0: Initial hidden state of shape (N, H)
  - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
  - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
  - b: Biases of shape (4H,)
  
  Returns a tuple of:
  - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
  - cache: Values needed for the backward pass.
  """
  h, cache = None, None
  #############################################################################
  # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
  # You should use the lstm_step_forward function that you just defined.      #
  #############################################################################
  N, T, D = x.shape

  cache = []

  B = (D-F)/S+1
  _, H = h0.shape
  H = H / B
  h = np.zeros((N, T, B*H))
  prev_h = h0
  prev_c = np.zeros((N, B*H))
  for t in xrange(T):
    prev_k = np.zeros((N, H))
    cc_t = []
    for b in xrange(B):
      # Get inputs
      xtk = x[:,t,b*S:b*S+F] # NxF
      bdh = prev_h[:,b*H:(b+1)*H] # NxH, b = block
      bdc = prev_c[:,b*H:(b+1)*H] # NxH

      a = np.dot(xtk, Wx) + np.dot(bdh, Wh) + np.dot(prev_k, Wk) + b
      ai, af, ao, ag = np.array_split(a, 4, axis=1)
      i = sigmoid(ai)
      f = sigmoid(af)
      o = sigmoid(ao)
      g = np.tanh(ag)
      ctk = f * bdc + i * g
      tc = np.tanh(ctk)
      htk = o * tc
      cc_tk = (xtk, bdh, bdc, prev_k, Wx, Wh, Wk, i, f, o, g, tc)
      h[:,t,b*H:(b+1)*H] = htk
      prev_h[:,b*H:(b+1)*H] = htk
      prev_c[:,b*H:(b+1)*H] = ctk
      prev_k = htk
      cc_t.append(cc_tk)
    cache.append(cc_t)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  cache.append((D, F, S))
  return h, cache


def tflstm_backward_advance_origin(dh, cache):
  dx, dh0, dWx, dWh, dWk, db = None, None, None, None, None, None
  N, T, H = dh.shape  # H = H / B
  D, F, S = cache[-1]
  B = (D-F)/S+1
  H = H / B

  dx = np.zeros((N, T, D))
  dh0 = np.zeros((N, B*H))
  dWx = np.zeros((F, 4*H))
  dWh = np.zeros((H, 4*H))
  dWk = np.zeros((H, 4*H))
  db = np.zeros(4*H)

  dnext_ht = np.zeros((N, B*H))
  dnext_ct = np.zeros((N, B*H))

  for t in xrange(T-1, -1, -1): # t = T-1, T-2, ..., 0
    cc_t = cache[t]
    dnext_hk = np.zeros((N, H))
    for b in xrange(B-1, -1, -1): # b = B-1, B-2, ..., 0
      xtk, prev_ht, prev_c, prev_hk, Wx, Wh, Wk, itk, ftk, otk, gtk, tctk = cc_t[b]
      dhtk = dh[:,t,b*H:(b+1)*H] + dnext_ht[:,b*H:(b+1)*H] + dnext_hk
      dtctk = dhtk * otk
      dctk = dnext_ct[:,b*H:(b+1)*H]
      dctk = dctk + dtctk * (1-tctk**2)

      dgtk = dctk * itk
      dotk = dhtk * tctk
      dftk = dctk * prev_c
      ditk = dctk * gtk

      daitk = ditk * itk * (1-itk)
      daftk = dftk * ftk * (1-ftk)
      daotk = dotk * otk * (1-otk)
      dagtk = dgtk * (1-gtk**2)
      datk = np.hstack((daitk, daftk, daotk, dagtk))

      dxtk = np.dot(datk, Wx.T)
      dprev_ht = np.dot(datk, Wh.T)
      dprev_c = dctk * ftk
      dprev_hk = np.dot(datk, Wk.T)
      dnext_ht[:,b*H:(b+1)*H] = dprev_ht
      dnext_ct[:,b*H:(b+1)*H] = dprev_c
      dnext_hk = dprev_hk
      dx[:,t,b*S:b*S+F] += dxtk
      dWx += np.dot(xtk.T, datk)
      dWh += np.dot(prev_ht.T, datk)
      dWk += np.dot(prev_hk.T, datk)
      db += datk.sum(axis=0)
  dh0 = dnext_ht
  return dx, dh0, dWx, dWh, db



def tflstm_forward_origin_step(x, prev_h, prev_c, Wx, Wh, Wk, bias, F, S, H):
  N, D = x.shape
  B = (D-F)/S+1

  cache = []

  next_h = np.zeros((N, B*H))
  next_c = np.zeros((N, B*H))
  prev_k = np.zeros((N, H))
  for b in xrange(B):
    # Get inputs
    xtk = x[:,b*S:b*S+F] # NxF
    bdh = prev_h[:,b*H:(b+1)*H] # NxH, b = block
    bdc = prev_c[:,b*H:(b+1)*H] # NxH

    a = np.dot(xtk, Wx) + np.dot(bdh, Wh) + np.dot(prev_k, Wk) + bias
    ai, af, ao, ag = np.array_split(a, 4, axis=1)
    i = sigmoid(ai)
    f = sigmoid(af)
    o = sigmoid(ao)
    g = np.tanh(ag)
    ctk = f * bdc + i * g
    tc = np.tanh(ctk)
    htk = o * tc

    cc_tk = (xtk, bdh, bdc, prev_k, Wx, Wh, Wk, i, f, o, g, tc)
    next_h[:,b*H:(b+1)*H] = htk
    next_c[:,b*H:(b+1)*H] = ctk
    prev_k = htk
    cache.append(cc_tk)
  cache.append((D, F, S, H))
  return next_h, next_c, cache


def tflstm_backward_advance_origin_step(dnext_h, dnext_c, cache):
  N = dnext_h.shape[0]
  D, F, S, H = cache[-1]
  B = (D-F)/S+1

  dx = np.zeros((N, D))
  dh = np.zeros((N, B*H))
  dc = np.zeros((N, B*H))
  dWx = np.zeros((F, 4*H))
  dWh = np.zeros((H, 4*H))
  dWk = np.zeros((H, 4*H))
  db = np.zeros(4*H)

  dnext_hk = np.zeros((N, H))
  for b in xrange(B-1, -1, -1): # b = B-1, B-2, ..., 0
    xtk, prev_ht, prev_c, prev_hk, Wx, Wh, Wk, itk, ftk, otk, gtk, tctk = cache[b]
    dhtk = dnext_h[:,b*H:(b+1)*H] + dnext_hk
    dtctk = dhtk * otk
    dctk = dnext_c[:,b*H:(b+1)*H]
    dctk = dctk + dtctk * (1-tctk**2)

    dgtk = dctk * itk
    dotk = dhtk * tctk
    dftk = dctk * prev_c
    ditk = dctk * gtk

    daitk = ditk * itk * (1-itk)
    daftk = dftk * ftk * (1-ftk)
    daotk = dotk * otk * (1-otk)
    dagtk = dgtk * (1-gtk**2)
    datk = np.hstack((daitk, daftk, daotk, dagtk))

    dxtk = np.dot(datk, Wx.T)
    dprev_ht = np.dot(datk, Wh.T)
    dprev_c = dctk * ftk
    dprev_hk = np.dot(datk, Wk.T)
    dh[:,b*H:(b+1)*H] = dprev_ht
    dc[:,b*H:(b+1)*H] = dprev_c
    dnext_hk = dprev_hk
    dx[:,b*S:b*S+F] += dxtk
    dWx += np.dot(xtk.T, datk)
    dWh += np.dot(prev_ht.T, datk)
    dWk += np.dot(prev_hk.T, datk)
    db += datk.sum(axis=0)
  return dx, dh, dc, dWx, dWh, dWk, db


def tflstm_forward_origin2(x, h0, Wx, Wh, Wk, bias, F, S, H):
  N, T, D = x.shape
  B = (D-F)/S+1

  h = np.zeros((N, T, B*H))
  cache = []

  prev_h = h0
  prev_c = np.zeros((N, B*H))
  for t in xrange(T):
    h[:,t,:], nc, cc_t = tflstm_forward_origin_step(x[:,t,:], prev_h, prev_c, Wx, Wh, Wk, bias, F, S, H)
    prev_h, prev_c = h[:,t,:], nc
    cache.append(cc_t)
  cache.append((D, F, S, H))
  return h, cache


def tflstm_backward_advance_origin2(dh, cache):
  N, T, _ = dh.shape
  D, F, S, H = cache[-1]
  B = (D-F)/S+1

  dx = np.zeros((N, T, D))
  dh0 = np.zeros((N, B*H))
  dWx = np.zeros((F, 4*H))
  dWh = np.zeros((H, 4*H))
  dWk = np.zeros((H, 4*H))
  db = np.zeros(4*H)

  dnext_h = np.zeros((N, B*H))
  dnext_c = np.zeros((N, B*H))
  for t in xrange(T-1, -1, -1):
    dnext_h += dh[:,t,:]
    dnext_hk = np.zeros((N, H))
    dxt = np.zeros((N, D))
    dht = np.zeros((N, B*H))
    dct = np.zeros((N, B*H))
    cachet = cache[t]
    for b in xrange(B-1, -1, -1): # b = B-1, B-2, ..., 0
      xtk, prev_ht, prev_c, prev_hk, Wx, Wh, Wk, itk, ftk, otk, gtk, tctk = cachet[b]
      dhtk = dnext_h[:,b*H:(b+1)*H] + dnext_hk
      dtctk = dhtk * otk
      dctk = dnext_c[:,b*H:(b+1)*H]
      dctk = dctk + dtctk * (1-tctk**2)
  
      dgtk = dctk * itk
      dotk = dhtk * tctk
      dftk = dctk * prev_c
      ditk = dctk * gtk
  
      daitk = ditk * itk * (1-itk)
      daftk = dftk * ftk * (1-ftk)
      daotk = dotk * otk * (1-otk)
      dagtk = dgtk * (1-gtk**2)
      datk = np.hstack((daitk, daftk, daotk, dagtk))
  
      dxtk = np.dot(datk, Wx.T)
      dprev_ht = np.dot(datk, Wh.T)
      dprev_c = dctk * ftk
      dprev_hk = np.dot(datk, Wk.T)
      dht[:,b*H:(b+1)*H] = dprev_ht
      dct[:,b*H:(b+1)*H] = dprev_c
      dnext_hk = dprev_hk
      dxt[:,b*S:b*S+F] += dxtk
      dWx += np.dot(xtk.T, datk)
      dWh += np.dot(prev_ht.T, datk)
      dWk += np.dot(prev_hk.T, datk)
      db += datk.sum(axis=0)
    dnext_h, dnext_c = dht, dct
    dx[:,t,:] = dxt
  dh0 = dnext_h
  return dx, dh0, dWx, dWh, dWk, db


# pp = peephole
def tflstm_forward_origin_step_pp(x, prev_h, prev_c, Wx, Wh, Wk, bias, pi, pf, po, F, S, H):
  N, D = x.shape
  B = (D-F)/S+1

  cache = []

  next_h = np.zeros((N, B*H))
  next_c = np.zeros((N, B*H))
  prev_k = np.zeros((N, H))
  for b in xrange(B):
    # Get inputs
    xtk = x[:,b*S:b*S+F] # NxF
    bdh = prev_h[:,b*H:(b+1)*H] # NxH, b = block
    bdc = prev_c[:,b*H:(b+1)*H] # NxH

    a = np.dot(xtk, Wx) + np.dot(bdh, Wh) + np.dot(prev_k, Wk) + bias
    ai, af, ao, ag = np.array_split(a, 4, axis=1)
    i = sigmoid(ai + bdc * pi)
    f = sigmoid(af + bdc * pf)
    g = np.tanh(ag)
    ctk = f * bdc + i * g
    o = sigmoid(ao + ctk * po)
    tc = np.tanh(ctk)
    htk = o * tc

    cc_tk = (xtk, bdh, bdc, prev_k, Wx, Wh, Wk, pi, pf, po, i, f, o, g, tc, ctk)
    next_h[:,b*H:(b+1)*H] = htk
    next_c[:,b*H:(b+1)*H] = ctk
    prev_k = htk
    cache.append(cc_tk)
  cache.append((D, F, S, H))
  return next_h, next_c, cache


def tflstm_backward_advance_origin_step_pp(dnext_h, dnext_c, cache):
  N = dnext_h.shape[0]
  D, F, S, H = cache[-1]
  B = (D-F)/S+1

  dx = np.zeros((N, D))
  dh = np.zeros((N, B*H))
  dc = np.zeros((N, B*H))
  dWx = np.zeros((F, 4*H))
  dWh = np.zeros((H, 4*H))
  dWk = np.zeros((H, 4*H))
  db = np.zeros(4*H)
  dpi = np.zeros(H)
  dpf = np.zeros(H)
  dpo = np.zeros(H)

  dnext_hk = np.zeros((N, H))
  for b in xrange(B-1, -1, -1): # b = B-1, B-2, ..., 0
    xtk, prev_ht, prev_c, prev_hk, Wx, Wh, Wk, pi, pf, po, itk, ftk, otk, gtk, tctk, ctk = cache[b]
    dhtk = dnext_h[:,b*H:(b+1)*H] + dnext_hk

    dotk = dhtk * tctk
    daotk = dotk * otk * (1-otk)

    dtctk = dhtk * otk
    dctk = dnext_c[:,b*H:(b+1)*H]
    dctk = dctk + dtctk * (1-tctk**2) + daotk * po

    dgtk = dctk * itk
    dftk = dctk * prev_c
    ditk = dctk * gtk

    daitk = ditk * itk * (1-itk)
    daftk = dftk * ftk * (1-ftk)
    dagtk = dgtk * (1-gtk**2)
    datk = np.hstack((daitk, daftk, daotk, dagtk))

    dxtk = np.dot(datk, Wx.T)
    dprev_ht = np.dot(datk, Wh.T)
    dprev_c = dctk * ftk + daitk * pi + daftk * pf
    dprev_hk = np.dot(datk, Wk.T)
    dh[:,b*H:(b+1)*H] = dprev_ht
    dc[:,b*H:(b+1)*H] = dprev_c
    dnext_hk = dprev_hk
    dx[:,b*S:b*S+F] += dxtk
    dWx += np.dot(xtk.T, datk)
    dWh += np.dot(prev_ht.T, datk)
    dWk += np.dot(prev_hk.T, datk)
    db += datk.sum(axis=0)
    dpi += np.sum(daitk * prev_c, axis=0)
    dpf += np.sum(daftk * prev_c, axis=0)
    dpo += np.sum(daotk * ctk, axis=0)
  return dx, dh, dc, dWx, dWh, dWk, db, dpi, dpf, dpo


def tflstm_forward_origin2_pp(x, h0, Wx, Wh, Wk, bias, pi, pf, po, F, S, H):
  N, T, D = x.shape
  B = (D-F)/S+1

  h = np.zeros((N, T, B*H))
  cache = []

  prev_h = h0
  prev_c = np.zeros((N, B*H))
  for t in xrange(T):
    h[:,t,:], nc, cc_t = tflstm_forward_origin_step_pp(x[:,t,:], prev_h, prev_c, Wx, Wh, Wk, bias, pi, pf, po, F, S, H)
    prev_h, prev_c = h[:,t,:], nc
    cache.append(cc_t)
  cache.append((D, F, S, H))
  return h, cache


def tflstm_backward_pp(dh, cache):
  N, T, _ = dh.shape
  D, F, S, H = cache[-1]
  B = (D-F)/S+1

  dx = np.zeros((N, T, D))
  dh0 = np.zeros((N, B*H))
  dWx = np.zeros((F, 4*H))
  dWh = np.zeros((H, 4*H))
  dWk = np.zeros((H, 4*H))
  db = np.zeros(4*H)
  dpi = np.zeros(H)
  dpf = np.zeros(H)
  dpo = np.zeros(H)

  dnext_h = np.zeros((N, B*H))
  dnext_c = np.zeros((N, B*H))
  for t in xrange(T-1, -1, -1):
    dnext_h += dh[:,t,:]
    dnext_hk = np.zeros((N, H))
    dxt = np.zeros((N, D))
    dht = np.zeros((N, B*H))
    dct = np.zeros((N, B*H))
    cachet = cache[t]
    for b in xrange(B-1, -1, -1): # b = B-1, B-2, ..., 0
      xtk, prev_ht, prev_c, prev_hk, Wx, Wh, Wk, pi, pf, po, itk, ftk, otk, gtk, tctk, ctk = cachet[b]
      dhtk = dnext_h[:,b*H:(b+1)*H] + dnext_hk
  
      dotk = dhtk * tctk
      daotk = dotk * otk * (1-otk)

      dtctk = dhtk * otk
      dctk = dnext_c[:,b*H:(b+1)*H]
      dctk = dctk + dtctk * (1-tctk**2) + daotk * po

      dgtk = dctk * itk
      dftk = dctk * prev_c
      ditk = dctk * gtk
  
      daitk = ditk * itk * (1-itk)
      daftk = dftk * ftk * (1-ftk)
      dagtk = dgtk * (1-gtk**2)
      datk = np.hstack((daitk, daftk, daotk, dagtk))
  
      dxtk = np.dot(datk, Wx.T)
      dprev_ht = np.dot(datk, Wh.T)
      dprev_c = dctk * ftk + daitk * pi + daftk * pf
      dprev_hk = np.dot(datk, Wk.T)
      dht[:,b*H:(b+1)*H] = dprev_ht
      dct[:,b*H:(b+1)*H] = dprev_c
      dnext_hk = dprev_hk
      dxt[:,b*S:b*S+F] += dxtk
      dWx += np.dot(xtk.T, datk)
      dWh += np.dot(prev_ht.T, datk)
      dWk += np.dot(prev_hk.T, datk)
      db += datk.sum(axis=0)
      dpi += np.sum(daitk * prev_c, axis=0)
      dpf += np.sum(daftk * prev_c, axis=0)
      dpo += np.sum(daotk * ctk, axis=0)
    dnext_h, dnext_c = dht, dct
    dx[:,t,:] = dxt
  dh0 = dnext_h
  return dx, dh0, dWx, dWh, dWk, db, dpi, dpf, dpo


def tflstm_forward_origin2_pp_unfold(x, h0, Wx, Wh, Wk, bias, pi, pf, po, F, S, H):
  N, T, D = x.shape
  B = (D-F)/S+1

  h = np.zeros((N, T, B*H))
  cache = []

  prev_h = h0
  prev_c = np.zeros((N, B*H))
  for t in xrange(T):
    next_h = np.zeros((N, B*H))
    next_c = np.zeros((N, B*H))
    cachet = []
    # input of this part is : x[:, t, :], prev_h, prev_c
    prev_k = np.zeros((N, H))
    for b in xrange(B):
      # Get inputs
      xtk = x[:,t,b*S:b*S+F] # NxF
      bdh = prev_h[:,b*H:(b+1)*H] # NxH, b = block
      bdc = prev_c[:,b*H:(b+1)*H] # NxH
  
      a = np.dot(xtk, Wx) + np.dot(bdh, Wh) + np.dot(prev_k, Wk) + bias
      ag, ai, af, ao = np.array_split(a, 4, axis=1)
      i = sigmoid(ai + bdc * pi)
      f = sigmoid(af + bdc * pf)
      g = np.tanh(ag)
      ctk = f * bdc + i * g
      o = sigmoid(ao + ctk * po)
      tc = np.tanh(ctk)
      htk = o * tc
  
      cc_tk = (xtk, bdh, bdc, prev_k, Wx, Wh, Wk, pi, pf, po, i, f, o, g, tc, ctk)
      next_h[:,b*H:(b+1)*H] = htk
      next_c[:,b*H:(b+1)*H] = ctk
      prev_k = htk
      cachet.append(cc_tk)
    # output of this part is : next_h, next_c
    prev_h, prev_c = next_h, next_c
    h[:,t,:] = next_h
    cachet.append((D, F, S, H))
    cache.append(cachet)
  cache.append((D, F, S, H))
  return h, cache



# format these formulas like the Kaldi nnet1 way
def tflstm_backward_pp_advance(dh, cache):
  N, T, _ = dh.shape
  D, F, S, H = cache[-1]
  B = (D-F)/S+1

  dx = np.zeros((N, T, D))
  dh0 = np.zeros((N, B*H))
  dWx = np.zeros((F, 4*H))
  dWh = np.zeros((H, 4*H))
  dWk = np.zeros((H, 4*H))
  db = np.zeros(4*H)
  dpi = np.zeros(H)
  dpf = np.zeros(H)
  dpo = np.zeros(H)

  dnext_at = np.zeros((N, B*4*H))
  dnext_ctk = np.zeros((N, B*H))
  dnext_aitk = np.zeros((N, B*H))
  dnext_aftk = np.zeros((N, B*H))
  next_ftk = np.zeros((N, B*H))

  for t in xrange(T-1, -1, -1):
    cachet = cache[t]
    dnext_atk = np.zeros((N, 4*H))
    for b in xrange(B-1, -1, -1): # b = B-1, B-2, ..., 0
      xtk, prev_ht, prev_c, prev_hk, Wx, Wh, Wk, pi, pf, po, itk, ftk, otk, gtk, tctk, ctk = cachet[b]

      dhtk = dh[:,t,b*H:(b+1)*H] + np.dot(dnext_at[:,b*4*H:(b+1)*4*H], Wh.T) + np.dot(dnext_atk, Wk.T)
  
      dtctk = dhtk * otk
      dtctk = dtctk * (1-tctk**2)

      dotk = dhtk * tctk
      daotk = dotk * otk * (1-otk)

      dctk = dtctk
      dctk += dnext_ctk[:,b*H:(b+1)*H] * next_ftk[:,b*H:(b+1)*H] + dnext_aitk[:,b*H:(b+1)*H] * pi + dnext_aftk[:,b*H:(b+1)*H] * pf
      dctk += daotk * po

      dftk = dctk * prev_c
      daftk = dftk * ftk * (1-ftk)
      ditk = dctk * gtk
      daitk = ditk * itk * (1-itk)
      dgtk = dctk * itk
      dagtk = dgtk * (1-gtk**2)
      datk = np.hstack((dagtk, daitk, daftk, daotk))

      # buffer
      dnext_atk = datk
      dnext_at[:,b*4*H:(b+1)*4*H] = datk
      dnext_ctk[:,b*H:(b+1)*H] = dctk
      dnext_aitk[:,b*H:(b+1)*H] = daitk
      dnext_aftk[:,b*H:(b+1)*H] = daftk
      next_ftk[:,b*H:(b+1)*H] = ftk

      dx[:,t,b*S:b*S+F] += np.dot(datk, Wx.T)
      dWx += np.dot(xtk.T, datk)
      dWh += np.dot(prev_ht.T, datk)
      dWk += np.dot(prev_hk.T, datk)
      db += datk.sum(axis=0)
      dpi += np.sum(daitk * prev_c, axis=0)
      dpf += np.sum(daftk * prev_c, axis=0)
      dpo += np.sum(daotk * ctk, axis=0)
  # dh0 is not important in ASR
  for b in xrange(B-1, -1, -1):
    dh0[:,b*H:(b+1)*H] = np.dot(dnext_at[:,b*4*H:(b+1)*4*H], Wh.T)
  return dx, dh0, dWx, dWh, dWk, db, dpi, dpf, dpo


