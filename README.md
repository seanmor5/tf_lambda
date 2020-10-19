# Lambda Networks

This is a TensorFlow 2 implementation of a Lambda Layer from: [LambdaNetworks: Modeling Long-Range Interactions Without Attention](https://openreview.net/pdf?id=xTJEN-ggl1b). LambdaNetworks achieve SOTA on ImageNet. For now, this library provides a basic UNTESTED version of the Lambda Layer based off of this [repository](https://github.com/lucidrains/lambda-networks).

## Usage

```python
import tensorflow as tf

layer = LambdaLayer(
    dim = 32,       # channels going in
    dim_out = 32,   # channels out
    n = 64 * 64,    # number of input pixels (64 x 64 image)
    dim_k = 16,     # key dimension
    heads = 4,      # number of heads, for multi-query
    dim_u = 1       # 'intra-depth' dimension
)

x = tf.random.normal(shape=(1, 64, 64, 32))
layer(x)
```