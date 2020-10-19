from tf_lambda.layer import LambdaLayer

class LambdaConv(Layer):

  def __init__(self, *, channels_out = None, receptive_field = None, key_dim = 16, intra_depth_dim = 1, heads = 4):
    super(LambdaConv, self).__init__()
    self.channels_out = channels_out
    self.receptive_field = receptive_field
    self.key_dim = key_dim
    self.intra_depth_dim = intra_depth_dim
    self.heads = heads

  def build(self, input_shape):
    self.layer = LambdaLayer(dim_out = self.channels_out, dim_k = self.key_dim, n = input_shape[1] * input_shape[2])

  def call(self, x):
    return self.layer(x)