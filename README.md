# matconv2caffe
Converts a matconvnet .mat model to the Caffe framework.

**Version:** 0.1

It works for the following layers:

- [x] conv / fc
- [x] normalization (might have problems for `kappa != 1`)
- [x] pool
- [x] softmax
- [x] ReLU
- [ ] Dropout (not checked)

## Usage
`python matconv2caffe.py [OPTIONS] input.mat`

For instance:

`python matconv2caffe.py model_name.mat --caffe_prefix $HOME/git/caffe/ --output model_name`

