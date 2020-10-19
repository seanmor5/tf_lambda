import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tf-lambda",
    version="0.1.1",
    author="Sean Moriarity",
    author_email="smoriarity.5@gmail.com",
    description="A TensorFlow 2 implementation of LambdaNetworks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/seanmor5/tf_lambda",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)