# coding=utf-8
# Copyright 2018 The Google AI Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Lint as: python2, python3


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import json
import os
import random
import time

from albert import fine_tuning_utils
from albert import modeling
from albert import squad_utils
import six
import tensorflow.compat.v1 as tf

import collections

from tensorflow.contrib import cluster_resolver as contrib_cluster_resolver
from tensorflow.contrib import tpu as contrib_tpu

import grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from tensorflow.python_io import tf_record_iterator
from tensorflow.contrib.util import make_tensor_proto


# pylint: disable=g-import-not-at-top
if six.PY2:
    import six.moves.cPickle as pickle
else:
    import pickle
# pylint: enable=g-import-not-at-top


def process_inputs(input_data):
    
    start = time.time()
    eval_examples = squad_utils.read_squad_examples(input_data,is_training=False)
    eval_features = []
    
    eval_writer = squad_utils.FeatureWriter(filename=predict_file,
                                            is_training=False)
    
    def append_feature(feature):
        eval_features.append(feature)
        eval_writer.process_feature(feature)
    
    tokenizer = fine_tuning_utils.create_vocab(
      vocab_file=None,
      do_lower_case=True,
      spm_model_file='/Users/benediktgroever/albert/albert/30k-clean.model',
      hub_module=None)
    
    squad_utils.convert_examples_to_features(
            examples=eval_examples,
            tokenizer=tokenizer,
            max_seq_length=384,
            doc_stride=128,
            max_query_length=64,
            is_training=False,
            output_fn=append_feature,
            do_lower_case=True)
    
    eval_writer.close()
    
    return eval_examples, eval_features



predict_file = '/Users/benediktgroever/albert/albert/check.tfrecords'
eval_examples, eval_features = process_inputs('input_file.json')

hostport = "127.0.0.1:8500"
channel = grpc.insecure_channel(hostport)
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
model_request = predict_pb2.PredictRequest()
model_request.model_spec.name = 'albert'

all_results = []
batch_size = 8

record_iterator = tf_record_iterator(path=predict_file)
for string_record in record_iterator: 
    
    model_request.inputs['examples'].CopyFrom(
        make_tensor_proto(string_record,
            dtype=tf.string,
            shape=[batch_size])
    )
    
    result_future = stub.Predict.future(model_request, 30.0)
    result = result_future.result().outputs
    all_results.append(result)



def process_result(result):

    unique_id = int(result["unique_ids"].int64_val[0])
    start_top_log_probs = ([float(x) for x in result["start_top_log_probs"].float_val])
    start_top_index = [int(x) for x in result["start_top_index"].int_val]
    end_top_log_probs = ([float(x) for x in result["end_top_log_probs"].float_val])
    end_top_index = [int(x) for x in result["end_top_index"].int_val]
    cls_logits = float(result["cls_logits"].float_val[0])
    

    RawResultV2 = collections.namedtuple("RawResultV2",["unique_id", "start_top_log_probs", "start_top_index",
                                                        "end_top_log_probs", "end_top_index", "cls_logits"])
    
    formatted_result = squad_utils.RawResultV2(
                unique_id=unique_id,
                start_top_log_probs=start_top_log_probs,
                start_top_index=start_top_index,
                end_top_log_probs=end_top_log_probs,
                end_top_index=end_top_index,
                cls_logits=cls_logits)
    
    return formatted_result


for i in range(len(all_results)):
	all_results[i] = process_result(all_results[i])


def process_output(all_results,eval_examples,eval_features,input_file,n_best,n_best_size,max_answer_length):
    
    
    with tf.gfile.Open(input_file, "r") as reader:
        input_data = json.load(reader)["data"]
    
    result_dict = {}
    cls_dict = {}
    
    #From Flags
    start_n_top = 1
    end_n_top = 1
    
    output_dir = '/Users/benediktgroever/albert/albert'
    
    output_prediction_file = os.path.join(output_dir, "predictions.json")
    output_nbest_file = os.path.join(output_dir, "nbest_predictions.json")
    output_null_log_odds_file = os.path.join(output_dir, "null_odds.json")
        
    squad_utils.accumulate_predictions_v2(result_dict,cls_dict,eval_examples,eval_features,all_results,
                                          n_best_size,max_answer_length,start_n_top,end_n_top)
              
    all_predictions, all_nbest_json = squad_utils.write_predictions_v2(result_dict, cls_dict,
                         eval_examples, # all_examples
                         eval_features, # all_features
                         all_results,   # all_results
                         n_best_size,   # n_best_size
                         max_answer_length, #max_answer_length
                         output_prediction_file, # output_prediction_file,
                         output_nbest_file, # output_nbest_file
                         output_null_log_odds_file, # output_null_log_odds_file
                         None) # null_score_diff_threshold
    
    
    re = []
    for i in range(len(all_predictions)):
        id_ = input_data[0]['paragraphs'][0]['qas'][i]['id']
        if n_best:
            re.append(collections.OrderedDict({
                    "id": id_,
                    "question": input_data[0]['paragraphs'][0]['qas'][i]["question"],
                    "best_prediction": all_predictions[id_],
                    "n_best_predictions": all_nbest_json[id_]
            }))
        else:
            re.append(collections.OrderedDict({
                    "id": id_,
                    "question": input_data[0]['paragraphs'][0]['qas'][i]["question"],
                    "best_prediction": all_predictions[id_]
            }))
    return re


n_best=False
n_best_size=20
max_answer_length=30

re = process_output(all_results,eval_examples,eval_features,'input_file.json',
                    n_best,n_best_size,max_answer_length)

# example for multiple inputs
# https://towardsdatascience.com/deploying-bert-using-kubernetes-6ddca23caec5

# bert model deployed on Tensorflow example
# https://medium.com/@joyceye04/deploy-a-servable-bert-qa-model-using-tensorflow-serving-d848f9797d9

# aws
# https://docs.aws.amazon.com/elastic-inference/latest/developerguide/ei-tensorflow-python.html

# google cloud
# https://colab.research.google.com/github/tensorflow/tfx/blob/master/docs/tutorials/serving/rest_simple.ipynb#scrollTo=LU4GDF_aYtfQ

