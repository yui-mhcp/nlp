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

import logging
import tensorflow as tf

from loggers import timer
from models.nlu.base_nlu_model import BaseNLUModel
from models.nlu.nlu_utils import DEFAULT_MAX_OUTPUT_LENGTH, infer_to_str, is_valid_tokens

logger = logging.getLogger(__name__)

class BaseNLUGenerator(BaseNLUModel):
    def __init__(self, * args, max_output_length = 1024, pretrained = 'facebook/bart-large', ** kwargs):
        self.max_output_length = max_output_length

        self.show_input = kwargs.get('show_input', True)
        super(BaseNLUGenerator, self).__init__(* args, pretrained = pretrained, ** kwargs)
    
    @property
    def training_hparams(self):
        return super(BaseNLUGenerator, self).training_hparams(
            max_output_length = None, teacher_forcing_eval = True, eval_infer_config = {}, show_input = None
        )

    @property
    def is_encoder_decoder(self):
        return hasattr(self.model, 'decoder') and self.model.decoder is not None
    
    @property
    def input_signature(self):
        signature = super().input_signature
        
        if self.is_encoder_decoder: signature = signature + self.text_signature
        return signature
    
    @property
    def output_signature(self):
        signature = self.multi_text_signature
        if self.is_encoder_decoder:
            return signature
        return signature + (tf.TensorSpec(shape = (None, ), dtype = tf.int32), )

    @property
    def encoder(self):
        return self.model.encoder
    
    @property
    def decoder(self):
        return self.model.decoder

    @timer(name = 'inference', log_if_root = False)
    def infer(self,
              text,
              text_length   = None,
              training      = False,
              merge_multi_input = False,
              force_not_subsampling = False,
              return_attention = False,
              ** kwargs
             ):
        kwargs.setdefault('max_length', self.max_output_length)
        if text_length is None and isinstance(text, (list, tuple)) and len(text) == 2:
            text, text_length = text
        
        if not isinstance(text, (list, tuple)):
            if len(tf.shape(text)) == 1: text = tf.expand_dims(text, axis = 0)
            
            if text_length is None:
                text_length = tf.fill([tf.shape(text)[0]], tf.shape(text)[1])
            elif len(tf.shape(text_length)) == 0:
                text_length = tf.expand_dims(text_length, axis = 0)
            
            text = [text, text_length]
        
        kwargs.update(self._get_mag_config(
            text,
            is_call     = False,
            training    = training,
            merge_multi_input = merge_multi_input,
            force_not_subsampling   = force_not_subsampling,
            ** kwargs
        ))
        
        return self.model.infer(
            text,
            training                = training,
            return_attention        = return_attention,
            return_hidden_states    = False,
            return_mask             = False,
            ** kwargs
        )
    
    def decode_output(self, output, ** kwargs):
        return self.decode_text(output.tokens)
    
    def compile(self,
                loss        = 'TextLoss',
                loss_config = {},
                metrics     = ['TextAccuracy', 'F1'],
                metrics_config  = {},
                optimizer_config    = {'lr' : 1e-5},
                ** kwargs
               ):
        loss_config['pad_value']    = self.blank_token_idx
        metrics_config.update(self.default_metrics_config)
        
        super().compile(
            loss    = loss,
            metrics = metrics,
            loss_config = loss_config,
            metrics_config  = metrics_config,
            optimizer_config    = optimizer_config,
            ** kwargs
        )
    
    def tf_multi_format_output(self, data, ** kwargs):
        kwargs.setdefault('max_length', self.max_output_length)
        return super().tf_multi_format_output(data, ** kwargs)
    
    def get_output(self, data, inputs = None, ** kwargs):
        tokens, lengths = self.tf_multi_format_output(data, ** kwargs)
        
        return tokens, lengths

    def encode_data(self, data):
        inputs, outputs = super().encode_data(data)

        if not self.is_encoder_decoder and len(inputs) == 2:
            inp_tokens, inp_length = inputs
            out_tokens, out_length = outputs
            
            tokens  = tf.concat([inp_tokens[:-1], out_tokens[1:]], axis = -1)
            
            inputs  = (tokens, inp_length + out_length - 2)
            outputs = (tokens, out_length - 1, inp_length - 1)

        return inputs, outputs
    
    def filter_output(self, outputs):
        """ Check `is_valid_tokens` for information """
        return is_valid_tokens(outputs[0], max_length = self.max_output_length)
    
    def augment_data(self, inputs, outputs):
        inp_tokens, inp_length = inputs[:2]
        
        inp_tokens, inp_length = self.augment_text(inp_tokens, inp_length)
        
        return (inp_tokens, inp_length) + inputs[2:], outputs
    
    def preprocess_data(self, inputs, outputs):
        if self.is_encoder_decoder:
            out_tokens, out_length = outputs
            out_in_tokens, out_in_length    = out_tokens[..., :-1], out_length - 1
            
            if len(tf.shape(out_in_tokens)) == 3:
                out_in_tokens, out_in_length = out_in_tokens[:, 0], out_in_length[:, 0]
            
            return inputs + (out_in_tokens, out_in_length), (out_tokens[..., 1:], out_length - 1)
        
        inp_tokens, inp_lengths = inputs
        out_tokens, out_lengths, skip_length = outputs
        return (
            (inp_tokens[:, :-1], inp_lengths - 1),
            (out_tokens[:, 1:], out_lengths, skip_length - 1)
        )
    
    def get_dataset_config(self, ** kwargs):
        inp_signature, out_signature = self.input_signature, self.output_signature
        if self.is_encoder_decoder: inp_signature = inp_signature[:-2]
        
        kwargs.update({
            'batch_before_map'  : True,
            'padded_batch'      : True,
            'pad_kwargs'        : {
                'padded_shapes'     : (
                    tuple([tuple(sign.shape)[1:] for sign in inp_signature]),
                    tuple([tuple(sign.shape)[1:] for sign in out_signature])
                ),
                'padding_values'    : (
                    tuple([self.blank_token_idx, 0] * (len(inp_signature) // 2)),
                    tuple([self.blank_token_idx] + [0] * (len(out_signature) - 1))
                )
            }
        })
        
        return super().get_dataset_config(** kwargs)

    def eval_step(self, batch):
        inputs, target = batch
        
        if self.teacher_forcing_eval:
            y_pred = self(inputs, training = False)
        else:
            if self.is_encoder_decoder: inputs = inputs[:-2]
            y_pred = self.infer(inputs, training = False, ** self.eval_infer_config).tokens

        return self.update_metrics(target, y_pred)

    @timer
    def predict_with_target(self, batch, epoch = None, step = None, prefix = None, 
                            directory = None, n_pred = 5, ** kwargs):
        inputs, output = batch
        inputs  = [inp[:n_pred] for inp in inputs]
        outputs = [out[:n_pred] for out in output]
        
        out_tokens, out_length  = outputs[:2]
        infer_inputs    = inputs[:-2] if self.is_encoder_decoder else inputs
        
        pred    = self(inputs, training = False, ** kwargs)
        infer   = self.infer(
            infer_inputs,
            max_length  = tf.shape(out_tokens)[-1],
            early_stopping  = False,
            ** {** self.eval_infer_config, ** kwargs}
        )
        
        pred_text   = self.decode_text(pred)
        infer_text  = self.decode_text(infer.tokens)
        
        input_text  = self.decode_text(inputs[0]) if self.show_input else None
        target_text = self.decode_text(out_tokens)
        
        preds = []
        for i in range(len(target_text)):
            preds.append("Prediction {} / {} :\n{}  Target     : {}\n  Prediction : {}\n{}".format(
                i + 1, len(target_text),
                "" if input_text is None else "  Input      : {}\n".format(input_text[i]),
                target_text[i],
                pred_text[i],
                '' if infer_text is None else infer_to_str(infer_text[i], infer.score[i], indent = 2)
            ))
        
        logger.info("\n".join(preds))
        
    def get_config(self, * args, ** kwargs):
        config = super().get_config(* args, ** kwargs)
        config['max_output_length'] = self.max_output_length
        return config