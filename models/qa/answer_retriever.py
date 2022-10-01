# Copyright (C) 2022 Quentin L. & yui-mhcp project's author. All rights reserved.
# Licenced under the Affero GPL v3 Licence (the "Licence").
# you may not use this file except in compliance with the License.
# See the "LICENCE" file at the root of the directory for the licence information.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf

from models.nlu.nlu_utils import find_index
from models.nlu.base_nlu_model import BaseNLUModel
from custom_architectures.transformers_arch.bert_arch import BertQA

class AnswerRetriever(BaseNLUModel):
    def __init__(self, * args, pretrained = 'bert-base-uncased',** kwargs):
        super().__init__(* args, pretrained = pretrained, ** kwargs)
    
    def _build_model(self, pretrained, ** kwargs):
        super()._build_model(
            model = BertQA.from_pretrained(pretrained, return_attention = False, ** kwargs)
        )
    
    @property
    def input_signature(self):
        return (
            tf.TensorSpec(shape = (None, None), dtype = tf.int32),  # text (tokens ids)
            tf.TensorSpec(shape = (None,), dtype = tf.int32),       # text length
            tf.TensorSpec(shape = (None, None), dtype = tf.int32)   # token type (0 = question, 1 = context)
        )
    
    @property
    def output_signature(self):
        return (
            tf.TensorSpec(shape = (None, ), dtype = tf.int32),  # Start idx
            tf.TensorSpec(shape = (None, ), dtype = tf.int32)   # End idx
        )
        
    def compile(self, loss = 'QARetrieverLoss', ** kwargs):
        super().compile(loss = loss, ** kwargs)
    
    def decode_output(self, output, inputs, ** kwargs):
        start, end = tf.argmax(output[0][0], axis = -1), tf.argmax(output[1][0], axis = -1)
        return self.decode_text(inputs[0][0, start : end], ** kwargs)

    def get_input(self, data, ** kwargs):
        encoded_text, token_types = self.tf_format_input(data, return_types = True, ** kwargs)
        
        return (encoded_text, len(encoded_text), token_types)

    def get_output(self, data, inputs = None, ** kwargs):
        if inputs is None: inputs = self.get_input(data)
        encoded_text = inputs[0]
        
        encoded_output  = self.tf_format_output(data, ** kwargs)[0][1:-1]

        start_idx = find_index(encoded_text, encoded_output)
        
        return start_idx, start_idx + len(encoded_output)

    def filter_output(self, outputs):
        return outputs[0] >= 0
    
    def get_dataset_config(self, ** kwargs):
        kwargs.update({
            'batch_before_map'  : True,
            'padded_batch'      : True,
            'pad_kwargs'        : {
                'padded_shapes'     : (
                    ((None,), (), (None,)), ((), ())
                ),
                'padding_values'    : ((self.blank_token_idx, 0, 0), (0, 0))
            }
        })
        
        return super().get_dataset_config(** kwargs)
