# matconv2caffe
Converts a matconvnet .mat model to the Caffe framework.

**Version:** 0.1

It works for the following layers:

- [*] conv / fc
- [*] normalization (might have problems for `kappa != 1`)
- [*] pool
- [*] softmax
- [*] ReLU
- [ ] Dropout (not checked)

## Usage
`python matconv2caffe.py [OPTIONS] input.mat`

