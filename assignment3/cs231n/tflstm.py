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


