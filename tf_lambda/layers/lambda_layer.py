import tensorflow as tf
from tensorflow import einsum
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, Softmax, Conv3D
from einops import rearrange

class LambdaLayer(Layer):

  def __init__(self, *, dim_k, n = None, r = None, heads = 4, dim_out = None, dim_u = 1):
    super(LambdaLayer, self).__init__()

    self.u = dim_u
    self.heads = heads

    assert (dim_out % heads) == 0
    dim_v = dim_out // heads

    self.to_q = Conv2D(filters = dim_k * heads, kernel_size = (1, 1), use_bias=False)
    self.to_k = Conv2D(filters = dim_k * dim_u, kernel_size = (1, 1), use_bias=False)
    self.to_v = Conv2D(filters = dim_v * dim_u, kernel_size = (1, 1), use_bias=False)

    self.norm_q = BatchNormalization()
    self.norm_v = BatchNormalization()

    self.local_contexts = r is not None

    if self.local_contexts:
      assert (r % 2) == 1, 'Receptive kernel size should be odd.'
      self.pad_fn = lambda x: tf.pad(x, tf.constant([[0, 0], [r // 2, r // 2], [r // 2, r // 2], [0, 0]]))
      self.pos_conv = Conv3D(filters = dim_k, kernel_size = (1, r, r))
      self.flatten = tf.keras.layers.Flatten()
    else:
      assert n is not None, 'You must specify the total sequence length (h x w)'
      self.pos_emb = self.add_weight(name='position_embed', shape=(n, n, dim_k, dim_u))

  def call(self, x):
    # For verbosity and understandings sake
    batch_size, height, width, channels, u, heads = *x.shape, self.u, self.heads
    b, hh, ww, c, u, h = batch_size, height, width, channels, u, heads

    q = self.to_q(x)
    k = self.to_k(x)
    v = self.to_v(x)

    q = self.norm_q(q)
    v = self.norm_v(v)

    q = rearrange(q, 'b hh ww (h k) -> b h (hh ww) k', h = h)
    k = rearrange(k, 'b hh ww (k u) -> b u (hh ww) k', u = u)
    v = rearrange(v, 'b hh ww (v u) -> b u (hh ww) v', u = u)

    k = tf.nn.softmax(k, axis=-1)

    lambda_c = einsum('b u m k, b u m v -> b k v', k, v)
    Y_c = einsum('b h n k, b k v -> b n h v', q, lambda_c)

    if self.local_contexts:
      v = rearrange(v, 'b u (hh ww) v -> b v hh ww u', hh = hh, ww = ww)
      # We need to add explicit padding across the batch dimension
      lambda_p = tf.map_fn(self.pad_fn, v)
      lambda_p = self.pos_conv(lambda_p)
      lambda_p = tf.reshape(lambda_p, (lambda_p.shape[0], lambda_p.shape[1], lambda_p.shape[2] * lambda_p.shape[3], lambda_p.shape[4]))
      Y_p = einsum('b h n k, b v n k -> b n h v', q, lambda_p)
    else:
      lambda_p = einsum('n m k u, b u m v -> b n k v', self.pos_emb, v)
      Y_p = einsum('b h n k, b n k v -> b n h v', q, lambda_p)

    Y = Y_c + Y_p
    out = rearrange(Y, 'b (hh ww) h v -> b hh ww (h v)', hh = hh, ww = ww)

    return out