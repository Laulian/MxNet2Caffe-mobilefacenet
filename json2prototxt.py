import sys
import argparse
import json
from prototxt_basic import *

parser = argparse.ArgumentParser(description='Convert MXNet jason to Caffe prototxt')
parser.add_argument('--mx-json',     type=str, default='./mxnet-model/model_mobilefacenet-symbol.json')
parser.add_argument('--cf-prototxt', type=str, default='./caffe-model/mxnet2caffe_mobilefacenet.prototxt')
args = parser.parse_args()

with open(args.mx_json) as json_file:    
  jdata = json.load(json_file)

with open(args.cf_prototxt, "w") as prototxt_file:
  for i_node in range(0,len(jdata['nodes'])):
    node_i    = jdata['nodes'][i_node]
    if str(node_i['op']) == 'null' and str(node_i['name']) != 'data':
      continue
    if str(node_i['op']) == '_copy' or str(node_i['op']) == '_minus_scalar' or str(node_i['op']) == '_mul_scalar' or str(node_i['op']) == 'Dropout':
      continue
    
    print('{}, \top:{}, name:{} -> {}'.format(i_node,node_i['op'].ljust(20),
                                        node_i['name'].ljust(30),
                                        node_i['name']).ljust(20))
    info = node_i
    
    info['top'] = info['name']
    info['bottom'] = []
    info['params'] = []
    for input_idx_i in node_i['inputs']:
      input_i = jdata['nodes'][input_idx_i[0]]
      if str(input_i['op']) == '_mul_scalar':
        info['bottom'].append('data')
      elif str(input_i['op']) == 'Dropout':
        current_input = input_i['inputs']
        current_input = current_input[0]
        input_idx = jdata['nodes'][current_input[0]]
        info['bottom'].append(str(input_idx['name']))
      elif str(input_i['op']) != 'null' or (str(input_i['name']) == 'data'):
        info['bottom'].append(str(input_i['name']))
      if str(input_i['op']) == 'null':
        #if str(input_i['name']) == 'fc1_weight':
         # print("fullyconnect : ",input_i['name'])
          #info['params'].append('pre_fc1_weight')
        #else:
        info['params'].append(str(input_i['name']))
        if not str(input_i['name']).startswith(str(node_i['name'])):
          print('           use shared weight -> %s'% str(input_i['name']))
          info['share'] = True
    write_node(prototxt_file, info)

