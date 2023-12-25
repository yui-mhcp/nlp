# Copyright (C) 2023-now yui-mhcp project's author. All rights reserved.
# Licenced under the Affero GPL v3 Licence (the "Licence").
# you may not use this file except in compliance with the License.
# See the "LICENCE" file at the root of the directory for the licence information.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import pandas as pd
import tensorflow as tf

from loggers import timer
from models.nlu.base_nlu_model import BaseNLUModel

logger = logging.getLogger(__name__)

DEFAULT_MAX_OUTPUT_LENGTH   = 1024

class BaseNLUGenerator(BaseNLUModel):
    output_signature    = BaseNLUModel.text_signature
    
    def __init__(self,
                 * args,
                 prompt = None,
                 pretrained = 'facebook/bart-large',
                 max_output_length = 1024,
                 ** kwargs
                ):
        self.prompt = prompt
        self.max_output_length = max_output_length

        if self.prompt and '{prompt}' not in kwargs.get('input_format', '{text}'):
            raise RuntimeError('`prompt` is provided but not used in `input_format` !')

        super(BaseNLUGenerator, self).__init__(* args, pretrained = pretrained, ** kwargs)
        
        if not self.output_format:
            if self.use_multi_input: raise NotImplementedError()
            self.output_format = self.input_format
    
    @property
    def is_encoder_decoder(self):
        return getattr(self.model, 'decoder', None) is not None
    
    @property
    def is_nwp(self):
        return self.input_format == self.output_format and not self.use_multi_input
    
    @property
    def input_signature(self):
        signature = super().input_signature
        
        if not self.is_encoder_decoder: return signature
        return (signature, self.text_signature)

    @property
    def training_hparams(self):
        return super(BaseNLUGenerator, self).training_hparams(
            max_output_length   = None,
            teacher_forcing_eval    = True,
            eval_infer_config   = {},
            show_input  = None
        )

    @timer(name = 'inference', log_if_root = False)
    def infer(self, * args, ** kwargs):
        kwargs.setdefault('max_length', self.max_output_length)
        return super().infer(* args, ** kwargs)
    
    def compile(self, loss = 'TextLoss', metrics = ['TextAccuracy'], ** kwargs):
        kwargs.setdefault('loss_config', {}).update({
            'pad_value' : self.blank_token_idx, 'eos_value' : self.blank_token_idx
        })
        kwargs.setdefault('metrics_config', {}).update(self.default_metrics_config)
        kwargs.setdefault('optimizer_config', {}).setdefault('lr', 1e-4)
        
        super().compile(loss = loss, metrics = metrics, ** kwargs)

    def format_input(self, * args, prompt = None, ** kwargs):
        if prompt is None: prompt = self.prompt
        if prompt is not None: kwargs['prompt'] = prompt
        return super().format_input(* args, ** kwargs)
    
    def get_output(self, data = None, inputs = None, ** kwargs):
        if self.input_format == self.output_format and not self.use_multi_input: return inputs
        return super().get_output(data = data, inputs = inputs, ** kwargs)

    def encode_data(self, data):
        inputs, outputs = super().encode_data(data)

        if self.is_encoder_decoder:
            if not isinstance(outputs, tuple):
                inputs, outputs = (inputs, outputs[:-1]), outputs[1:]
            else:
                inputs, outputs = (inputs, outputs[0][:-1]), (outputs[0][1:], ) + outputs[1:]
        elif isinstance(inputs, tuple):
            raise NotImplementedError()
        elif isinstance(outputs, tuple):
            if self.sos_token: outputs = (outputs[0][1:], ) + outputs[1:]
            if self.eos_token: inputs  = inputs[:-1]
            inputs = tf.concat([inputs, outputs[0]], axis = -1)
        else:
            if self.sos_token: outputs = outputs[1:]
            if self.eos_token: inputs  = inputs[:-1]
            inputs = tf.concat([inputs, outputs], axis = -1)

        return inputs, outputs
    
    def filter_output(self, outputs):
        if isinstance(outputs, tuple): outputs = outputs[0]
        return tf.logical_and(
            tf.shape(outputs)[0] > 0, tf.shape(outputs)[-1] <= self.max_output_length
        )
    
    def get_dataset_config(self, ** kwargs):
        inp_signature, out_signature = self.input_signature, self.output_signature
        
        kwargs.update({
            'batch_before_map'  : True,
            'padded_batch'      : True,
            'pad_kwargs'        : {
                'padding_values'    : (
                    tf.nest.map_structure(lambda sign: self.blank_token_idx, inp_signature),
                    tf.nest.map_structure(
                        lambda sign: tf.constant(
                            0, dtype = sign.dtype
                        ) if sign.dtype != tf.int32 else self.blank_token_idx,
                        out_signature
                    )
                )
            }
        })
        
        return super().get_dataset_config(** kwargs)

    def eval_step(self, batch):
        inputs, target = batch
        
        if self.teacher_forcing_eval:
            y_pred = self(inputs, training = False)
        else:
            if self.is_encoder_decoder: inputs = inputs[:-1]
            y_pred = self.infer(inputs, training = False, ** self.eval_infer_config).tokens

        return self.update_metrics(target, y_pred)

    @timer
    def predict_with_target(self, batch, epoch = None, step = None, prefix = None, 
                            directory = None, n_pred = 5, ** kwargs):
        inputs, outputs = tf.nest.map_structure(lambda t: t[:n_pred], batch)
        
        out_tokens      = outputs[0] if isinstance(outputs, tuple) else outputs
        infer_inputs    = inputs[:-1] if self.is_encoder_decoder else inputs
        if len(infer_inputs) == 1: infer_inputs = infer_inputs[0]
        
        pred    = self(inputs, training = False, ** kwargs)
        infer   = self.infer(
            infer_inputs,
            max_length  = tf.shape(out_tokens)[-1],
            early_stopping  = False,
            ** {** self.eval_infer_config, ** kwargs}
        )
        
        pred_text   = self.decode_text(pred)
        infer_text  = self.decode_text(infer)
        
        input_text  = self.decode_text(inputs[0]) if self.show_input else None
        target_text = self.decode_text(out_tokens)
        
        preds = []
        for i in range(len(target_text)):
            preds.append("Prediction {} / {} :\n{}  Target     : {}\n  Prediction : {}\n{}".format(
                i + 1, len(target_text),
                "" if input_text is None else "  Input      : {}\n".format(input_text[i]),
                target_text[i],
                pred_text[i],
                infer_to_str(
                    infer_text[i], infer.scores[i], indent = 2
                ) if infer_text is not None else ''
            ))
        
        logger.info("\n".join(preds))
        
    def get_config(self, * args, ** kwargs):
        config = super().get_config(* args, ** kwargs)
        config.update({
            'prompt'    : self.prompt,
            'max_output_length' : self.max_output_length
        })
        return config
    
def infer_to_str(text, score, indent = 0):
    _indentation = ' ' * indent
    if not isinstance(text, (list, tuple)):
        return '{}Inference ({:.3f}) : {}'.format(_indentation, score, text)
    
    des = '{}Inference :'.format(_indentation)
    for j, (s, txt) in enumerate(zip(score, text)):
        des += '\n{}  #{} ({:.3f}) : {}'.format(_indentation, j, s, txt)
    return des
