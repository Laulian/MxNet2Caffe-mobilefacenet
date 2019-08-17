#coding:utf-8

import sys
import os
os.environ['GLOG_minloglevel'] = '2'
import numpy as np
import mxnet as mx
#insert your caffe path
sys.path.insert(0, 'E:/caffe-master/python')
import caffe
import cv2
import sklearn

def mxnet_get_model(ctx, image_size, model_str, layer):
  _vec = model_str.split(',')
  assert len(_vec)==2
  prefix = _vec[0]
  epoch = int(_vec[1])
  print('loading',prefix, epoch)
  sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
  all_layers = sym.get_internals()
  # print(all_layers)
  sym = all_layers[layer+'_output']
  model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
  model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
  model.set_params(arg_params, aux_params)
  model.aux_params = aux_params
  model.arg_params = arg_params
  return model

def mxnet_get_feature(img , mx_model, layer):
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img = np.transpose(img, (2, 0, 1))
  ctx = mx.gpu(0)
  image_size = (int(112), int(112))
  model = mxnet_get_model(ctx, image_size, mx_model, layer)
  input_blob = np.expand_dims(img, axis=0)
  data = mx.nd.array(input_blob)
  db = mx.io.DataBatch(data=(data,))
  model.forward(db, is_train=False)
  embedding = model.get_outputs()[0].asnumpy()
  return embedding

def mxnet_get_weights(img , mx_model, last_layer, layer):
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img = np.transpose(img, (2, 0, 1))
  ctx = mx.gpu(0)
  image_size = (int(112), int(112))
  model = mxnet_get_model(ctx, image_size, mx_model, last_layer)
  # extract weights
  weight = model.aux_params[layer].asnumpy()
  return weight

def caffe_get_feature(deploy, caffe_model, img, layer):
    caffe.set_mode_gpu()
    net = caffe.Net(deploy, caffe_model, caffe.TEST)

    # tensor preprocess
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_raw_scale('data', 0.0078125)
    transformer.set_mean('data', np.array((0.0078125*127.5, 0.0078125*127.5, 0.0078125*127.5)))
    im = transformer.preprocess('data', img)
    net.blobs['data'].data[...] = im
    net.forward()
    #list all layers's output feature map
    features = net.blobs.keys()
    #extract output feature
    feature = np.float64(net.blobs[layer].data[0])
    return feature

def caffe_get_weights(deploy, caffe_model, img, layer):
    caffe.set_mode_gpu()
    net = caffe.Net(deploy, caffe_model, caffe.TEST)

    # tensor preprocess
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_raw_scale('data', 0.0078125)
    transformer.set_mean('data', np.array((0.0078125*127.5, 0.0078125*127.5, 0.0078125*127.5)))
    im = transformer.preprocess('data', img)
    net.blobs['data'].data[...] = im
    net.forward()
    # list all layers's weights
    layers_weight = net.params.keys()
    #extract weights
    weight0 = net.params[layer][0].data
    weight1 = net.params[layer][1].data
    return weight0



if __name__ == "__main__":
    img = cv2.imread('Aaron_Eckhart_0001.jpg')
    # layer = 'conv_2_dw_conv2d'
    deploy = './caffe-model/mxnet2caffe_mobilefacenet.prototxt'
    caffe_model = './caffe-model/mxnet2caffe_mobilefacenet.caffemodel'
    mxnet_model_epoch = './mxnet-model/model_mobilefacenet,200'
    # check output feature of layer
    caffe_feature = caffe_get_feature(deploy, caffe_model, img, 'fc1')
    # print(caffe_feature)
    mxnet_feature = mxnet_get_feature(img, mxnet_model_epoch, 'fc1')
    # print(mxnet_feature)
    dist = np.sum(np.square(caffe_feature-mxnet_feature[0]))
    print("dist  =  %d\n"%dist)
    caffe_feature_norm = (np.sqrt(np.sum(np.square(caffe_feature))))
    mxnet_feature_norm = (np.sqrt(np.sum(np.square(mxnet_feature[0]))))
    sim = np.dot(caffe_feature, mxnet_feature[0].T)/caffe_feature_norm/mxnet_feature_norm
    print("sim  =  %d\n"%sim)

    #check weights of layer
    caffe_weights = caffe_get_weights(deploy, caffe_model, img, 'fc1')
    mxnet_weights = mxnet_get_weights(img, mxnet_model_epoch,'fc1', 'fc1_moving_mean')
    print(caffe_feature)
    print(mxnet_feature)

