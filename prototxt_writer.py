# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 10:18:38 2015

@author: prlz77
"""
nl = '\n'
def write_header(output, name, data, crops=1, channels=1, width=1, height=1):
    output.write('name: "' + name + '"' + nl)
    output.write('input: "' + data + '"' + nl)
    output.write('input_dim: ' + str(crops) + nl)
    output.write('input_dim: ' + str(channels) + nl)
    output.write('input_dim: ' + str(height) + nl)
    output.write('input_dim: ' + str(width) + nl)

def format_param(param, value):
    if isinstance(value, str):
        return param + ': "' + value + '"'
    else:
        return param + ': ' + str(value)

def encapsulate(typestr, params):
    for i in range(len(params)):
        params[i] = '  ' + params[i]
    return [ typestr  + ' {' ] + params + [ '}' ]
    
def write_conv_layer(output, name, bottom, num_output, kernel_size, stride, pad=0, group=False):
    block = [ format_param('name', name) ]
    block += [ format_param('type', 'Convolution') ]
    block += [ format_param('bottom', bottom) ]
    block += [ format_param('top',  name) ]
    
    layer_params = [ format_param('num_output', num_output) ]
    layer_params += [ format_param('kernel_size', kernel_size) ]
    layer_params += [ format_param('stride' , stride) ]
    if group:
        layer_params += [ format_param('group', 2)]
    
    if pad:
        layer_params += [ format_param('pad', pad) ]
    
    block += encapsulate('convolution_param', layer_params)
    
    block = encapsulate('layer', block)
    
    output.write(nl.join(block) + nl)
    
def write_pool_layer(output, name, bottom, pool, kernel_size, stride, pad=0):
    block = [ format_param('name', name) ]
    block += [ format_param('type', 'Pooling') ]
    block += [ format_param('bottom', bottom) ]
    block += [ format_param('top',  name) ]
    
    layer_params = [ 'pool: ' + pool ]
    layer_params += [ format_param('kernel_size', kernel_size) ]
    layer_params += [ format_param('stride' , stride) ]
    
    if pad:
        layer_params += [ format_param('pad', pad) ]
        
    block += encapsulate('pooling_param', layer_params)
    
    block = encapsulate('layer', block)
    
    output.write(nl.join(block) + nl)
    
def write_relu_layer(output, name, bottom):
    block = [ format_param('name', name) ]
    block += [ format_param('type', "ReLU") ]
    block += [ format_param('bottom', bottom) ]
    block += [ format_param('top', bottom) ]
    
    block = encapsulate('layer', block)
    
    output.write(nl.join(block) + nl)

def write_norm_layer(output, name, bottom, local_size, alpha, beta):
    block = [ format_param('name', name) ]
    block += [ format_param('type', 'LRN') ]
    block += [ format_param('bottom', bottom) ]
    block += [ format_param('top', name) ]
    
    layer_params = [ format_param('local_size', local_size) ]
    layer_params += [ format_param('alpha', alpha) ]
    layer_params += [ format_param('beta', beta) ]
    
    block += encapsulate('lrn_param', layer_params)
    block = encapsulate('layer', block)
    
    output.write(nl.join(block) + nl)
    
def write_dropout_layer(output, name, bottom, dropout_ratio):
    block = [ format_param('name', name) ]
    block += [ format_param('type', 'Dropout') ]
    block += [ format_param('bottom', bottom) ]
    block += [ format_param('top', bottom) ]
    
    layer_params = [ format_param('dropout_ratio', dropout_ratio) ]
    
    block += encapsulate('dropout_param', layer_params)
    block = encapsulate('layer', block)
    
    output.write(nl.join(block) + nl)
    
def write_fc_layer(output, name, bottom, num_output):
    block = [ format_param('name', name) ]
    block += [ format_param('type', 'InnerProduct')]
    block += [ format_param('bottom', bottom) ]
    block += [ format_param('top', name) ]
    
    layer_params = [ format_param('num_output', num_output) ]
    
    block += encapsulate('inner_product_param', layer_params)
    block = encapsulate('layer', block)
    
    output.write(nl.join(block) + nl)
    
def write_softmax_layer(output, name, bottom):
    block = [ format_param('name', name) ]
    block += [ format_param('type', 'Softmax')]
    block += [ format_param('bottom', bottom) ]
    block += [ format_param('top', name) ]   
    block = encapsulate('layer', block)
    output.write(nl.join(block) + nl)

if __name__=='__main__':
    import sys
    f = sys.stdout
    write_header(f, 'prova')
    write_conv_layer(f, 'conv1', 'data', 128, 5, 1)
    write_relu_layer(f, 'relu1', 'conv1')
    write_pool_layer(f, 'pool1', 'conv1', pool='MAX', kernel_size=3, stride=2)
    write_norm_layer(f, 'norm1', 'pool1', 5, 0.5, 0.5)
    write_fc_layer(f, 'fc6', 'pool5', 4096)
    write_dropout_layer(f, 'drop6', 'fc6', dropout_ratio=0.5)
    write_softmax_layer(f, 'prob', 'fc8')