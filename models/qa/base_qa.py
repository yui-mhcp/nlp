# Copyright (C) 2022 Quentin Langlois & yui-mhcp project's author. All rights reserved.
# Licenced under the Affero GPL v3 Licence (the "Licence").
# you may not use this file except in compliance with the License.
# See the "LICENCE" file at the root of the directory for the licence information.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pandas as pd
import tensorflow as tf

from models.interfaces.base_text_model import BaseTextModel
from models.weights_converter import partial_transfer_learning

DEFAULT_MAX_INPUT_LENGTH = 512

def find_index(text, answer, start_idx = 0):
    idx = -1
    possible_starts = tf.where(text == answer[0])
    if len(tf.shape(possible_starts)) == 2:
        for i in tf.cast(tf.squeeze(possible_starts, axis = 1), tf.int32):
            tokens = text[i : i + len(answer)]
            if len(tokens) == len(answer) and tf.reduce_all(tokens == answer):
                idx = i
                break

    return idx

class BaseQAModel(BaseTextModel):
    def __init__(self,
                 lang,
                 
                 input_format   = ['{question}', '{context}'],
                 output_format  = '{answer}',
                 
                 max_input_length   = DEFAULT_MAX_INPUT_LENGTH,
                 use_fixed_length_input = False,
                 
                 pretrained = None,
                 
                 ** kwargs
                ):
        if use_fixed_length_input: raise NotImplementedError()
        
        self._init_text(lang = lang, ** kwargs)
        
        self.input_format   = input_format
        self.output_format  = output_format
        self.max_input_length   = max_input_length
        self.use_fixed_length_input = use_fixed_length_input
        
        kwargs.setdefault('pretrained_name', pretrained)
        super().__init__(pretrained = pretrained, ** kwargs)
                
        if hasattr(self.model, '_build'): self.model._build()
    
    @property
    def training_hparams(self):
        return super().training_hparams(max_input_length = None, ** self.training_hparams_text)
    
    
    def __str__(self):
        des = super().__str__()
        des += self._str_text()
        des += "- Input format : {}\n".format(self.input_format)
        des += "- Output format : {}\n".format(self.output_format)
        return des
    
    def format_input(self, question = None, context = None, title = None, ** kwargs):
        return self.text_encoder.format(
            self.input_format, question = question, context = context, title = title, ** kwargs
        )
    
    def format_output(self, question = None, context = None, title = None, answer = None,
                      ** kwargs):
        return self.text_encoder.format(
            self.output_format, question = question, context = context, title = title,
            answer = answer, ** kwargs
        )
    
    def tf_format_input(self, data):
        encoded_text, token_types = tf.py_function(
            self.format_input,
            [data.get('question', ''), data.get('context', ''), data.get('title', '')],
            Tout = [tf.int32, tf.int32]
        )
        encoded_text.set_shape([None])
        token_types.set_shape([None])
        
        return encoded_text, token_types
    
    def tf_format_output(self, data):
        encoded_text, token_types = tf.py_function(
            self.format_output,
            [data.get('question', ''), data.get('context', ''), data.get('title', ''), data.get('answers', '')],
            Tout = [tf.int32, tf.int32]
        )
        encoded_text.set_shape([None])
        token_types.set_shape([None])
        
        return encoded_text, token_types
    
    def get_input(self, data):
        tokens, _ = self.tf_format_input(data)
        
        return (tokens, len(tokens))
    
    def get_output(self, data, inputs = None):
        tokens, _ = self.tf_format_output(data)
        
        return (tokens, len(tokens))
    
    def encode_data(self, data):
        inputs = self.get_input(data)

        outputs = self.get_output(data, inputs)

        return inputs, outputs
    
    def filter_data(self, inputs, outputs):
        return inputs[1] <= self.max_input_length
    
    def get_config(self, * args, ** kwargs):
        config = super().get_config(* args, ** kwargs)
        config.update({
            ** self.get_config_text(),
            'input_format'  : self.input_format,
            'output_format' : self.output_format,
            
            'max_input_length'  : self.max_input_length,
            'use_fixed_length_input'    : self.use_fixed_length_input
        })
        return config
        
