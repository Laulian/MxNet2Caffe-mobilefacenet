import sys, argparse
import os
os.environ['GLOG_minloglevel'] = '2'
import mxnet as mx
#insert your caffe path
sys.path.insert(0, 'E:/caffe-master/python')
import caffe

parser = argparse.ArgumentParser(description='Convert MXNet model to Caffe model')
parser.add_argument('--mx-model',    type=str, default='./mxnet-model/model_mobilefacenet')
parser.add_argument('--mx-epoch',    type=int, default=200)
parser.add_argument('--cf-prototxt', type=str, default='./caffe-model/mxnet2caffe_mobilefacenet.prototxt')
parser.add_argument('--cf-model',    type=str, default='./caffe-model/mxnet2caffe_mobilefacenet.caffemodel')
args = parser.parse_args()

# ------------------------------------------
# Load
#_, arg_params, aux_params = mx.model.load_checkpoint(args.mx_model, args.mx_epoch)
syms, arg_params, aux_params = mx.model.load_checkpoint(args.mx_model, args.mx_epoch)
print("-------load mxnet model successful")
net = caffe.Net(args.cf_prototxt, caffe.TEST)
print("-------load caffe prototxt success")
# ------------------------------------------
# Convert
all_keyss = arg_params.keys() + aux_params.keys()
all_keyss.sort()
all_keys = all_keyss
#all_keys[78]=all_keyss[79]
#all_keys[79]='pre_fc1_bias'
print('----------------------------------\n')
print('ALL KEYS IN MXNET:')
print(all_keys)
print('%d KEYS' %len(all_keys))

print('----------------------------------\n')
print('VALID KEYS:')
for i_key,key_i in enumerate(all_keys):

  try:
    if 'data' is key_i:
      pass
    # elif 'fc1_weight' in key_i:
    #   key_caffe = key_i.replace('fc1_weight', 'fc1')
    #   net.params[key_caffe][0].data.flat = arg_params[key_i].asnumpy().flat
    elif '_weight' in key_i:
      key_caffe = key_i.replace('_weight','')
      # if 'fc' in key_i:
      #   print key_i
      #   print arg_params[key_i].shape
      #   key_caffe = 'pre_fc1'
      #   print net.params[key_caffe][0].data.shape
      net.params[key_caffe][0].data.flat = arg_params[key_i].asnumpy().flat      
    elif '_bias' in key_i:
      key_caffe = key_i.replace('_bias','')
      net.params[key_caffe][1].data.flat = arg_params[key_i].asnumpy().flat   
    elif '_gamma' in key_i and 'relu' not in key_i:
      key_caffe = key_i.replace('_gamma','_scale')
      net.params[key_caffe][0].data.flat = arg_params[key_i].asnumpy().flat
    # TODO: support prelu
    elif '_gamma' in key_i:   # for prelu
      key_caffe = key_i.replace('_gamma','')
      assert (len(net.params[key_caffe]) == 1)
      net.params[key_caffe][0].data.flat = arg_params[key_i].asnumpy().flat
    elif '_beta' in key_i:
      key_caffe = key_i.replace('_beta','_scale')
      net.params[key_caffe][1].data.flat = arg_params[key_i].asnumpy().flat    
    elif '_moving_mean' in key_i and 'fc1' not in key_i:
      key_caffe = key_i.replace('_moving_mean','')
      net.params[key_caffe][0].data.flat = aux_params[key_i].asnumpy().flat
      net.params[key_caffe][2].data[...] = 1
    elif '_moving_var' in key_i and 'fc1' not in key_i:
      key_caffe = key_i.replace('_moving_var','')
      net.params[key_caffe][1].data.flat = aux_params[key_i].asnumpy().flat
      net.params[key_caffe][2].data[...] = 1
    elif '_moving_mean' in key_i:
      key_caffe = key_i.replace('_moving_mean', '')
      net.params[key_caffe][0].data.flat = aux_params[key_i].asnumpy().flat
      net.params[key_caffe][2].data[...] = 1
    elif '_moving_var' in key_i:
      key_caffe = key_i.replace('_moving_var', '')
      net.params[key_caffe][1].data.flat = aux_params[key_i].asnumpy().flat
      net.params[key_caffe][2].data[...] = 1
    else:
      sys.exit("Warning!  Unknown mxnet:{}".format(key_i))
  
    print("% 3d | %s -> %s, initialized." 
           %(i_key, key_i.ljust(40), key_caffe.ljust(30)))
    
  except KeyError:
    print("\nError!  key error mxnet:{}".format(key_i))

# fc1 = mx.sym.BatchNorm(data=key_i, fix_gamma=True, eps=2e-5, momentum=0.9, name='fc1')
# ------------------------------------------

# Finish
net.save(args.cf_model)
print("\n- Finished.\n")

