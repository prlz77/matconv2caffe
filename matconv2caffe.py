# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 13:01:03 2015

@author: prlz77
@version: 0.1
"""
from scipy.io import loadmat
import numpy as np
from bilinear_interpolate import *
import os
import sys
import argparse

parser = argparse.ArgumentParser(description='Convert matconvnet into caffe model.')

parser.add_argument('input', type=str, help='Input .mat model.')
parser.add_argument('--output', '-o', type=str, help='Output .caffemodel.')
parser.add_argument('--extract_avg', '-a', action='store_true', 
                    help='Wether to extract the average image (numpy + protobin)')
parser.add_argument('--caffe_prefix', '-p', type=str, default='',
                    help='Caffe installation path to import caffe protobuf libraries')
parser.add_argument('--verbose', '-v', action='store_true')

args = parser.parse_args()

VERBOSE=args.verbose
INPUT=args.input

if args.output:
    OUTPUT=args.output
else:
    OUTPUT=INPUT.split('.')[0] + '.caffemodel'

PROTOTXT= INPUT.split('.')[0] + '.prototxt'

if args.caffe_prefix:
    CAFFE_PATH=args.caffe_prefix
    sys.path.append(os.path.join(CAFFE_PATH, 'python/caffe/proto'))

def log(message):
    if VERBOSE:
        print message

import caffe_pb2
import prototxt_writer as protow

prototxt = open(PROTOTXT, 'w')

print 'Reading...'
log('reading matconvnet file ' + INPUT)
matconv_params = loadmat(INPUT)

blob = caffe_pb2.NetParameter()

net_name = os.path.basename(INPUT).split('.')[0]
blob.name = net_name

matconv_layers = matconv_params['layers'][0]

caffe_layers = blob.layer

image_size = matconv_params['normalization']['imageSize'].item()[0]

#caffe_layers.add()
#caffe_layers[0].name = 'data'
blob.input.append("data")
protow.write_header(prototxt, net_name, 
                    'data',
                    crops=1, 
                    channels=int(image_size[-1]), 
                    width=int(image_size[-2]),
                    height=int(image_size[-3]))

log('Copying layers')
first_conv = True

last_top = 'data'
last_channels = 3

print 'Transfering weights...'
for i in range(matconv_layers.size):
    caffe_layers.add()
    layer = matconv_layers[i]
    layer_name = str(layer['name'].item().item())
    caffe_layers[-1].name = layer_name
    log(caffe_layers[-1].name)
    
    layer_type = layer['type'].item().item()
    
    if layer_type == 'conv':
        layer_size = layer['weights'].item()[0][0].shape[::-1]
        if layer_size[-2] == 1 or 'fc' in layer_name:
            protow.write_fc_layer(prototxt, layer_name, last_top, layer_size[0])
        else:
            group = layer_size[1] == (last_channels / 2)
            protow.write_conv_layer(prototxt, layer_name, last_top, 
                                    num_output=layer_size[0], 
                                    kernel_size=int(layer_size[2]),
                                    stride=int(layer['stride'][0][0][0][0]),
                                    pad=int(layer['pad'][0][0][0][0]),
                                    group=group )           
            
        last_top = layer_name
    elif layer_type == 'relu':
        protow.write_relu_layer(prototxt, layer_name, last_top)
    elif layer_type == 'normalize':
        params = layer['param'][0][0][0]
        protow.write_norm_layer(prototxt, layer_name, last_top, 
                                int(params[0]), 
                                params[2]*params[0],
                                params[3])
        last_top = layer_name
    elif layer_type == 'pool':
        method = layer['method'].item().item().upper()
        protow.write_pool_layer(prototxt, layer_name, last_top, 
                                pool=method, 
                                kernel_size=int(layer['pool'][0][0][0][0]),
                                stride=int(layer['stride'][0][0][0][0]),
                                pad=int(layer['pad'][0][0][0][0]))
        last_top = layer_name
    elif layer_type == 'dropout':
        protow.write_dropout_layer(prototxt, layer_name, last_top,
                                   layer['rate'][0][0][0])
    elif layer_type == 'softmax':
        protow.write_softmax_layer(prototxt, layer_name, last_top)
        last_top = layer_name
    # Weights
    try:
        weights =  matconv_layers[i]['weights'].item()[0][0].copy()
        caffe_layers[-1].blobs.add()
        weights = weights.transpose((3, 2, 0, 1))
        if first_conv and layer_size[1] == 3:
            weights = weights[:, ::-1, :, :] # to bgr
        if 'conv' in layer_name:
            caffe_layers[-1].blobs[0].shape.dim[:] = layer_size[:]
            
        else:
            caffe_layers[-1].blobs[0].shape.dim[:] = [layer_size[0], np.prod(layer_size[1:])]
        caffe_layers[-1].blobs[0].data[:] = weights.astype(float).flat
        last_channels = layer_size[0]
    except Exception:
        log('No weights to copy in this layer')
    # Bias     
    try:
        bias =  matconv_layers[i]['weights'].item()[0][1].copy()
        caffe_layers[-1].blobs.add()
        caffe_layers[-1].blobs[1].shape.dim.append(layer_size[0])
        caffe_layers[-1].blobs[1].data[:] = bias.astype(float).flat      
    except:
        log('No biases to copy in this layer')

print 'Writting to ' + OUTPUT + ' , ' + PROTOTXT
log('Writting to file ' + OUTPUT)
with open(OUTPUT, 'w') as output:
    output.write(blob.SerializeToString())

log('Closing prototxt ' + PROTOTXT)
prototxt.close()

if args.extract_avg:
    log('Extracting average image')
    border = matconv_params['normalization']['border'].item().squeeze()
    complete_image_size = image_size[:-1] + border

    avg = matconv_params['normalization']['averageImage'].item()
    avg = avg.transpose()
    avg = avg[::-1, :, :]
    
    x = np.linspace(0, image_size[0]-1, complete_image_size[0])
    y = np.linspace(0, image_size[1]-1, complete_image_size[1])
    x, y = np.meshgrid(x, y, sparse=False, indexing='xy')
    avg2 = bilinear_interpolate(avg, x, y)
    
    mean = caffe_pb2.BlobProto()
    mean.data[:] = avg2.flat

    with open(INPUT.split('.')[0] + '.binaryproto', 'w') as output:
        output.write(mean.SerializeToString())
    
    with open(INPUT.split('.')[0] + '.npy', 'w') as output:
        np.save(output, avg2)



