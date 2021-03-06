# Lambda Networks

This is a TensorFlow 2 implementation of a Lambda Layer from: [LambdaNetworks: Modeling Long-Range Interactions Without Attention](https://openreview.net/pdf?id=xTJEN-ggl1b). LambdaNetworks achieve SOTA on ImageNet. For now, this library provides a basic UNTESTED version of the Lambda Layer based off of this [repository](https://github.com/lucidrains/lambda-networks).

## Usage

### Lambda Layer

Global Context

```python
import tensorflow as tf
from tf_lambda.layers.lambda_layer import LambdaLayer

layer = LambdaLayer(
    dim_out = 32,   # channels out
    n = 64 * 64,    # number of input pixels (64 x 64 image)
    dim_k = 16,     # key dimension
    heads = 4,      # number of heads, for multi-query
    dim_u = 1       # 'intra-depth' dimension
)

x = tf.random.normal(shape=(1, 64, 64, 32))
layer(x)
```

Localized Context

```python
layer = LambdaLayer(
    dim_out = 32,
    r = 23,         # the receptive field for relative positional encoding (23 x 23)
    dim_k = 16,
    heads = 4,
    dim_u = 4
)

x = tf.random.normal(shape=(1, 64, 64, 32))
layer(x)
```
### Lambda Conv

Just a wrapper around the Lambda Layer. Builds it specifically for image data. The only thing you need to specify is the number of channels to output.

```python
from tf_lambda.layers.lambda_conv import LambdaConv

layer = LambdaConv(32)

x = tf.random.normal(shape=(1, 64, 64, 32))
layer(x)
```