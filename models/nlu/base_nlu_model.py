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

import os
import glob
import logging
import numpy as np
import pandas as pd
import tensorflow as tf

from loggers import timer
from utils.thread_utils import Pipeline
from models.nlu.nlu_utils import DEFAULT_MAX_INPUT_LENGTH, _get_expected_keys, _get_key_mapping, is_valid_tokens
from models.interfaces.base_text_model import BaseTextModel, _get_key_value_format
from custom_architectures.transformers_arch import get_pretrained_transformer
from custom_architectures.transformers_arch.mag_wrapper import MAGWrapper

logger      = logging.getLogger(__name__)
time_logger = logging.getLogger('timer')

class BaseNLUModel(BaseTextModel):
    _alternative_keys   = [
        ('answer', 'answers'),
        ('context', 'paragraph', 'text', 'contexts', 'paragraphs', 'texts'),
        ('title', 'titles')
    ]
    _multi_alternative_keys = [
        ('answers', 'answer'),
        ('contexts', 'paragraphs', 'texts', 'context', 'paragraph', 'text'),
        ('titles', 'title')
    ]
    
    def __init__(self,
                 lang,
                 
                 input_format,
                 output_format,
                 
                 input_multi_format = None,
                 subsample_input    = False,
                 multi_input_offset = -1,
                 
                 use_multi_input    = True,
                 
                 split_multi_input  = False,
                 split_key          = None,
                 max_split_sentences    = -1,
                 max_sentence_length    = -1,
                 
                 max_total_length   = -1,
                 sort_by_length     = False,
                 
                 max_input_texts    = -1,
                 input_select_mode  = 'start',
                 
                 max_input_length   = DEFAULT_MAX_INPUT_LENGTH,
                 max_multi_input_length = -1,
                 use_fixed_length_input = False,
                 
                 pretrained = None,
                 
                 ** kwargs
                ):
        """
            Constructor for a simple interface for NLU models.
            
            Arguments :
                - lang  / text_encoder  : classic config for `BaseTextModel`
                
                - input_format  : text format for the model's input
                - output_format : text format for the model's output
                
                # all these arguments are only relevant if `input_multi_format` is provided. They are forwarded to `utils.text.text_processing.filter_texts` to filter which texts to keep
                # All these configuration (except `input_multi_format`) are `training_hparams`, meaning that you can re-specify them when training / evaluation / prediction
                - input_multi_format    : text format for multi-inputs
                - use_multi_input       : whether to use multiple inputs or not (only relevant if `input_multi_format` is provided)
                - split_multi_input     : whether to split the `multi_format` when longer than `max_input_length`
                - split_key             : the key on which to split `multi_format`
                - max_split_sentences   : maximum number of sentences per paragraph
                - max_sentence_length   : maximum length for splitted parts (default to `max_input_length`)
                - sort_by_length        : whether to sort by length or not when `max_total_length > 0`
                - max_total_length      : used to determine the maximal length for `input_multi_format`
                - max_input_texts       : maximal number of texts for multiple inputs
                - input_select_mode     : the way to select input texts when they are too many (according to `max_input_texts`)
                
                - max_input_length  : maximum size for the input text
                - max_multi_input_length    : maximum length for the multi-input texts. If `split_multi_context = True`, this constraint is applied on the paragraph's lengths (i.e. the sum of its sentences' lengths).
                If `split_multi_input = False`, it is set by default to `max_input_length`
                - use_fixed_length_input    : whether to pad the input text to a fixed size (not supported yet)
            
            The `{input / output}_keys` properties are inferred based on `{input / output}_format`, based on the keys between "{}".
            
            Example for a french-to-english translation model :
            Constructor's arguments :
                - input_format   : '{fr}'
                - output_format  : '{en}'
                - input_keys     : ['fr']
                - output_keys    : ['en']
            In this case, when you call `get_input(data)`, it will call `text_encoder.format('{fr}', fr = text)`
            where `text` is either the given `data` or, if `data` is a dict / pd.Series, `text = data['fr']`
            
            Note that, in this case, `input_format` only contains 1 item and thus `input_keys` also contains 1 item. 
            In a sentiment analysis scenario, `input_format = ['{title}', '{text}'] # equivalent to '{title}{sep_token}{text}'` which allows to concatenate the text's title and the text.
            In this case, `input_keys = ['title', 'text']` (`{sep_token}` is not taken into account as it is a special token, not an expected data key).
            Note that, in this case, `data` must be a list of 2 items or a dict / pd.Series containing the keys 'title' and 'text'. If one of those keys is not in `data`, it will be set to the empty string.
            
            The `input_multi_format` is used to give multiple inputs at once. It comes from the `Memory Augmented Generator : a new approach for question-answering` paper which uses the question and multiple paragraphs as input
            To reproduce this phenomenon, use `input_format = {question}` and `input_multi_format = {title}{sep_token}{context}`.
            In this case, `get_input()` returns a 4-items tuple `(q_tokens, q_length, c_tokens, c_lengths)` where `q_tokens.shape == (None, )` and `c_tokens.shape == (None, None)`
        """
        if use_fixed_length_input: raise NotImplementedError()
        
        self._init_text(lang = lang, ** kwargs)
        
        self.input_format   = input_format
        self.output_format  = output_format
        
        self.input_multi_format = input_multi_format
        self.subsample_input    = subsample_input
        self.multi_input_offset = multi_input_offset
        
        self.use_multi_input    = use_multi_input
        self.split_multi_input  = split_multi_input
        self.split_key          = split_key
        self.max_split_sentences    = max_split_sentences
        self.max_sentence_length    = max_sentence_length
        self.sort_by_length     = sort_by_length
        self.max_total_length   = max_total_length
        self.max_input_texts    = max_input_texts
        self.input_select_mode  = input_select_mode
        
        self.max_input_length   = max_input_length
        self.max_multi_input_length = max_multi_input_length
        self.use_fixed_length_input = use_fixed_length_input
        
        input_keys  = _get_expected_keys(input_format)
        multi_keys  = _get_expected_keys(input_multi_format)
        output_keys = _get_expected_keys(output_format)
        
        self.input_keys     = None
        self.multi_keys     = None
        self.output_kess    = None
        
        if self.split_multi_input:
            if not self.split_key:
                if len(multi_keys) > 1:
                    logger.warning('Multiple keys in `multi_format` but no `split_key` : taking by default the last one ({})'.format(multi_keys[-1]))
                self.split_key = multi_keys[-1]
            elif self.split_key not in multi_keys:
                raise ValueError('`split_key` ({}) is not in `multi_format` expected keys ({}) !'.format(self.split_key, multi_keys))
        
        self.input_mapping  = _get_key_mapping(input_keys, self._alternative_keys)
        self.multi_mapping  = _get_key_mapping(multi_keys, self._multi_alternative_keys if self._multi_alternative_keys else self._alternative_keys)
        self.output_mapping = _get_key_mapping(output_keys, self._alternative_keys)
        
        self.min_multi_input_length = -1
        self.merge_multi_input      = None
        
        if isinstance(pretrained, str):
            kwargs.setdefault('pretrained_name', pretrained)
        super(BaseNLUModel, self).__init__(pretrained = pretrained, ** kwargs)
        
        if hasattr(self.model, 'set_tokens'): self.model.set_tokens(** self.model_tokens)
    
    def _build_model(self, pretrained = None, ** kwargs):
        if self.input_multi_format: kwargs['wrapper'] = MAGWrapper
            
        if pretrained is not None:
            super(BaseNLUModel, self)._build_model(
                model = get_pretrained_transformer(pretrained, ** kwargs)
            )
        else:
            super(BaseNLUModel, self)._build_model(** kwargs)
    
    def init_train_config(self, ** kwargs):
        super(BaseNLUModel, self).init_train_config(** kwargs)

        max_model_types = None
        if 'encoder_max_types' in self.model.hparams:
            max_model_types = self.model.hparams.encoder_max_types
        elif 'max_types' in self.model.hparams:
            max_model_types = self.model.hparams.max_types
        
        if max_model_types is not None:
            if max_model_types > 0:
                if self.max_input_texts > 0:
                    if self.max_input_texts > max_model_types:
                        logger.warning('Restricting max_input_texts to {} instead of {}'.format(
                            max_model_types, self.max_negatives
                        ))
                    self.max_input_texts = min(
                        self.max_input_texts, max_model_types
                    )
                else:
                    self.max_input_texts = max_model_types

    @property
    def is_encoder_decoder(self):
        return False
    
    @property
    def input_signature(self):
        signature = ()
        if self.input_format: signature += (self.text_signature, )
        if self.input_multi_format:
            if self.use_multi_input or self.split_multi_input:
                signature += (self.multi_text_signature, )
            else:
                signature += (self.text_signature, )
        return signature
    
    @property
    def training_hparams(self):
        multi_input_config = {} if self.input_multi_format is None else {
            'multi_keys'    : None,
            'use_multi_input'   : None,
            'max_input_texts'   : None,
            'input_select_mode' : 'random',

            'split_multi_input' : None,
            'max_split_sentences'   : None,
            'max_sentence_length'   : None,
            
            'sort_by_length'    : None,
            'max_total_length'  : None,
            
            'min_multi_input_length'    : None,
            'merge_multi_input' : None,
            'max_multi_input_length'    : None
        }
        return super(BaseNLUModel, self).training_hparams(
            input_keys  = None,
            output_keys = None,
            max_input_length    = None,
            ** self.training_hparams_text,
            ** multi_input_config
        )
    
    @property
    def default_metrics_config(self):
        return {
            'pad_value' : self.blank_token_idx,
            'eos_value' : self.eos_token_idx,
            'decode_fn' : lambda text: self.decode_text(text, remove_tokens = True)
        }
    
    @property
    def in_batch_merging(self):
        if self.input_multi_format is None:
            raise ValueError('`self.input_multi_format` is None')
        if self.merge_multi_input is None:
            return not self.use_multi_input
        return self.merge_multi_input
    
    @property
    def subsampling_factor(self):
        if self.input_multi_format is None:
            raise ValueError('`self.input_multi_format` is None')
        return self.model.hparams.encoder_subsampling_step

    def __str__(self):
        des = super(BaseNLUModel, self).__str__()
        des += self._str_text()
        if self.input_format:
            des += "- Input format : {}\n".format(self.input_format)
        if self.input_multi_format:
            des += "- Multi input format : {}\n".format(self.input_multi_format)
        if self.output_format:
            des += "- Output format : {}\n".format(self.output_format)
        if self.input_multi_format:
            des += "- Split multi input (key : {}) : {}\n".format(
                self.split_key, self.split_multi_input
            )
            if self.split_multi_input:
                des += "- Max sentences per split : {}\n".format(self.max_split_sentences)
            des += "- # of memory layers : {}\n".format(len(self.encoder.memory_layers))
            des += "- # of embedding layers : {}\n".format(len(self.encoder.embedding_layers))
            des += "- Subsampling factor : {}\n".format(self.subsampling_factor)
            des += "- Subsampling mode : {}\n".format(
                self.model.hparams.encoder_subsampling_mode
            )
        
        des += "- Max input length : {}\n".format(self.max_input_length)
        if self.input_multi_format and self.split_multi_input:
            des += "- Max sentence length : {}\n".format(self.max_sentence_length)
        
        return des
    
    def _get_mag_config(self,
                        inputs,
                        is_call,
                        training    = False,
                        merge_multi_input   = False,
                        force_not_subsampling  = False,
                        ** kwargs
                       ):
        if self.input_multi_format is None: return {}
        if self.is_encoder_decoder and is_call: inputs = inputs[0]
        
        n_multi_input = len(inputs)
        if self.input_format is not None: n_multi_input -= 1
        
        not_subsampling = force_not_subsampling if self.subsample_input else (
            ([True] if self.input_format is not None else [])
            + [force_not_subsampling] * n_multi_input
        )
        positional_offset = -1 if self.multi_input_offset <= 0 else (
            ([-1] if self.input_format is not None else [])
            + [self.multi_input_offset] * n_multi_input
        )
        
        prefix = '' if not self.is_encoder_decoder else 'encoder_'
        return {
            '{}merge_embeddings'.format(prefix) : merge_multi_input or (training and self.in_batch_merging),
            '{}force_not_subsampling'.format(prefix) : not_subsampling,
            '{}positional_offset'.format(prefix) : positional_offset
        }
        
    @timer(name = 'prediction', log_if_root = False)
    def call(self,
             inputs,
             training       = False,
             merge_multi_input  = False,
             force_not_subsampling  = False,
             ** kwargs
            ):
        kwargs.update(self._get_mag_config(
            inputs,
            is_call     = True,
            training    = training,
            merge_multi_input = merge_multi_input,
            force_not_subsampling   = force_not_subsampling,
        ))
        
        return self.model(inputs, training = training, ** kwargs)

    def infer(self, inputs, training = False, ** kwargs):
        return self(inputs, training = training, ** kwargs)
    
    def decode_output(self, output, ** kwargs):
        return self.decode_text(output.tokens if hasattr(output, 'tokens') else output, ** kwargs)
    
    def format_input(self, * args, ** kwargs):
        return self.format_text(self.input_format, * args, ** kwargs)
    
    def format_output(self, * args, ** kwargs):
        return self.format_text(self.output_format, * args, ** kwargs)
    
    def tf_format_data(self,
                       text_format,
                       data,
                       
                       split    = False,
                       multi_format = False,
                       
                       default_keys = None,
                       default_mapping = None,
                       
                       ** kwargs
                      ):
        if default_mapping is not None:
            kwargs['keys_mapping'] = default_mapping
            kwargs.setdefault('keys', default_keys if default_keys is not None else default_mapping)
        
        if multi_format:
            if split:
                return self.tf_multi_split_and_format(
                    text_format, self.split_key, data, ** kwargs
                )
            return self.tf_multi_format(text_format, data, ** kwargs)
        elif split:
            return self.tf_split_and_format_text(
                text_format, self.split_key, data, ** kwargs
            )
        return self.tf_format_text(text_format, data, ** kwargs)
    
    def tf_format_input(self, data, ** kwargs):
        """ Returns the formatted single input (according to `self.input_format`) """
        return self.tf_format_data(
            self.input_format,
            data,
            split   = False,
            multi_format    = False,
            default_keys    = self.input_keys,
            default_mapping = self.input_mapping,
            ** kwargs
        )
    
    def tf_multi_format_input(self, data, ** kwargs):
        """ Returns the formatted multi inputs (according to `self.input_multi_format`) """
        if self.input_multi_format is None:
            raise RuntimeError('`self.input_multi_format` is None')
        
        kwargs.setdefault('max_texts',      self.max_input_texts)
        kwargs.setdefault('select_mode',    self.input_select_mode)
        
        kwargs.setdefault('min_length',     self.min_multi_input_length)
        kwargs.setdefault('sort_by_length', self.sort_by_length)
        kwargs.setdefault('max_total_length',   self.max_total_length)
        
        if self.split_multi_input:
            kwargs.setdefault('max_length',     self.max_multi_input_length)
            kwargs.setdefault('max_sentences',  self.max_split_sentences)
            kwargs.setdefault(
                'max_sentences_length', self.max_sentence_length if self.max_sentence_length > 0 else self.max_input_length
            )
        else:
            kwargs.setdefault(
                'max_length', self.max_multi_input_length if self.max_multi_input_length > 0 else self.max_input_length
            )

        if isinstance(data, (dict, pd.Series)) and 'valid_idx' in data:
            kwargs.setdefault('required_idx', data['valid_idx'])

        return self.tf_format_data(
            self.input_multi_format,
            data,
            split   = self.split_multi_input,
            multi_format = self.use_multi_input,
            default_keys = self.multi_keys,
            default_mapping = self.multi_mapping,
            ** kwargs
        )
    
    def tf_format_output(self, data, multi_format = False, ** kwargs):
        """ Returns the formatted single input (according to `self.output_format`) """
        return self.tf_format_data(
            self.output_format,
            data,
            split   = False,
            multi_format    = multi_format,
            default_keys    = self.output_keys,
            default_mapping = self.output_mapping,
            ** kwargs
        )

    def tf_multi_format_output(self, data, ** kwargs):
        return self.tf_format_output(data, multi_format = True, ** kwargs)

    def get_input(self, data, ** kwargs):
        """
            Returns the encoded inputs according to `self.input_format` and `self.multi_input_format`
        """
        inputs = ()
        if self.input_format is not None:
            inputs += (self.tf_format_input(data, ** kwargs), )
        
        if self.input_multi_format is not None:
            inputs += (self.tf_multi_format_input(data, ** kwargs), )
        
        return inputs
    
    def get_output(self, data, inputs = None, ** kwargs):
        raise NotImplementedError()
    
    def encode_data(self, data):
        """ Returns `(self.get_input(data), self.get_output(data))` """
        inputs  = self.get_input(data)
        outputs = self.get_output(data, inputs = inputs)

        return inputs, outputs
    
    def filter_input(self, inputs):
        """ Check `is_valid_tokens` for information """
        if self.is_encoder_decoder: inputs = inputs[0]
        if not isinstance(inputs, tuple): inputs = (inputs, )
        
        valid = True

        if self.input_format is not None:
            valid = is_valid_tokens(inputs[0], max_length = self.max_input_length)
        if valid and self.input_multi_format is not None:
            max_multi_length = -1
            if self.split_multi_input: max_multi_length = self.max_sentence_length
            elif self.max_multi_input_length is not None: max_multi_length = self.max_multi_input_length
            else: max_multi_length = self.max_input_length
            valid = is_valid_tokens(inputs[-1], max_length = max_multi_length)
            
            if valid and self.split_multi_input and self.max_split_sentences > 0:
                valid = valid and tf.shape(inputs[-1])[0] <= self.max_split_sentences
            

            if self.max_input_texts > 0:
                valid = valid and tf.shape(inputs[-1])[0] <= self.max_input_texts * (
                    1 if not self.split_multi_input else max(self.max_split_sentences, 1)
                )
        
        return valid
    
    def filter_output(self, outputs):
        return True
    
    def filter_data(self, inputs, outputs):
        return self.filter_input(inputs) and self.filter_output(outputs)
    
    def get_pipeline(self,
                     batch_size = 1,
                     
                     metrics    = None,
                     
                     analyze_attention  = False,
                     
                     save       = False,
                     save_attention = False,
                     overwrite  = False,
                     directory  = None,
                     map_file   = 'map.json',
                     
                     ** kwargs
                    ):
        """
            Initializes the inference pipeline taking a `row` of data (valid `self.get_input` data) and performs the inference + decoding + (optional) metrics computation (for each generated candidate) and attention analysis (cf `analyze_last_attention`, introduced in the paper `Memory Augmented Generator : a new approach for question-answering`)
            
            Arguments :
                - batch_size    : `self.infer` batch_size
                
                - metrics   : (list of) metrics' names to compute for each candidate (adds the `metrics` key to each candidate)
                
                - analyze_attention : whether to call `self.analyze_last_attention` or not (adds the `attention_infos` key to each candidate and to the result)
                
                - save  : whether to save the output in `filename`
                - save_attention    : whether to save the attention in a `.npy` file (ignored if `analyze_attention = False`)
                - overwrite : whether to overwrite or not
                - directory : where to save the results (attentions are saved in `{directory}/attentions` sub-directory)
                - map_file  : the mapping filename
                
                - kwargs    : propagated to the `Pipeline` constructor
            
            Pipeline procedure :
                1) Creating an `id` field in data (if not already there) to identify the input
                    By default it is computed as the string representation of the key-value mapping of the input format's kes
                2) Encode the data (`self.get_input(row)`) and adds the `encoded` key
                3) Performs the model's inference (`self.infer(row['encoded']`) and adds the `output` key
                4) Decodes the `output` key (`self.decode_output`) and adds the `candidates` key (list of dict containing (at least) the `text` key)
                5) (optional) Computes metrics and adds the `metrics` key to each candidate
                6) (optional) Analyses attention (`self.analyze_last_attention`) and sets `attention_infos` keys (to the output + to each candidate for individual information)
        """
        def add_data_id(row, ** kwargs):
            """ Adds an `id` key to identify the input (row) to avoid re-computing duplicates """
            if 'id' not in row:
                mapping = {}
                if self.input_format:
                    keys, values = _get_key_value_format(
                        row, keys = self.input_keys, keys_mapping = self.input_mapping
                    )
                    mapping.update({k : v for k, v in zip(keys, values)})

                if self.input_multi_format:
                    keys, values = _get_key_value_format(
                        row, keys = self.multi_keys, keys_mapping = self.multi_mapping
                    )
                    mapping.update({k : v for k, v in zip(keys, values)})

                row['id'] = str(mapping)
            return row

        def encode(infos, ** kwargs):
            """ Adds the `encoded` key to `infos` """
            infos['encoded'] = self.get_input(infos, ** kwargs)
            return infos
        
        def inference(infos, ** kwargs):
            """ Performs inference and adds the `output` key """
            if isinstance(infos, list):
                pad_values  = self.get_dataset_config()['pad_kwargs']['padding_values'][0]
                encoded     = [info['encoded'] for info in infos]
                batch_inputs = tf.nest.map_structure(
                    lambda * inp: tf.cast(pad_batch(inp[:-1], pad_value = inp[-1])),
                    * encoded, pad_values
                )
            else:
                batch_inputs    = tf.nest.map_structure(
                    lambda t: tf.expand_dims(t, axis = 0), infos['encoded']
                )
            
            logger.info('Inputs shape : {} - config : {}'.format(
                [tuple(inp.shape) for inp in batch_inputs], kwargs
            ))
            
            return_attn   = kwargs.get('analyze_attention', analyze_attention)
            output = self.infer(
                batch_inputs, training = False, return_last_attention = return_attn, ** kwargs
            )
            
            if isinstance(infos, list):
                for b, info in enumerate(infos):
                    info['output'] = tf.nest.map_structure(
                        lambda o: o[b] if o is not None else o, output
                    )
            else:
                infos['output'] = tf.nest.map_structure(
                    lambda o: o[0] if o is not None else o, output
                )
            
            return infos
        
        def decode(infos, ** kwargs):
            """ Decodes `output` and sets the `candidates` key (list of dict with `text` key) """
            candidates = self.decode_output(
                infos['output'], inputs = infos['encoded'], data = infos, ** kwargs
            )
            scores = infos['output'].score if hasattr(infos['output'], 'score') else None
            if not isinstance(candidates, list):
                if scores is not None: scores = [scores]
                candidates = [candidates]
            
            results = []
            for i, cand in enumerate(candidates):
                if not isinstance(cand, dict): cand = {'text' : cand}
                if scores is not None: cand['score'] = scores[i]
                results.append(cand)
            
            infos['candidates'] = results
            return infos

        def attention_analyzis(infos, ** kwargs):
            def _add_attn_info(candidate, attn_weights):
                global_infos, attn_infos = self.analyze_last_attention(
                    infos['encoded'], attn_weights, ** kwargs
                )
                infos['attention_infos'] = global_infos
                candidate['attention_infos'] = attn_infos
            
            output = infos['output']
            if not hasattr(output, 'attention_weights'): return infos
            kwargs['save'] = kwargs.pop('save_attention', save_attention)
            kwargs['directory'] = None if not kwargs['save'] else attn_dir

            attn_weights = output.attention_weights
            if hasattr(output, 'tokens') and len(tf.shape(output.tokens)) == 2:
                for b, cand in enumerate(infos['candidates']):
                    _add_attn_info(
                        cand, tf.nest.map_structure(lambda attn: attn[b], attn_weights)
                    )
            else:
                _add_attn_info(
                    infos['candidates'][0], attn_weights
                )
            
            return infos
        
        def compute_metrics(infos, ** kwargs):
            """ Adds the `metrics` key to each `candidates` """
            def _get_metrics(target, pred):
                metrics.reset_states()
                metrics.update_state(target, pred)
                return {
                    name : val for name, val in zip(
                        metrics.metric_names, metrics.result().numpy()
                    )
                }
            
            target = self.get_output(infos)
            
            if self.filter_output(target):
                target  = tuple([tf.expand_dims(t[..., 1:], axis = 0) for t in target])
                output  = tf.nest.map_structure(
                    lambda o: tf.expand_dims(o, axis = 0) if o is not None else None, infos['output']
                )
                # If beam-search prediction, set metrics to each candidate
                if hasattr(output, 'tokens') and len(tf.shape(output.tokens)) == 3:
                    for b in range(tf.shape(output.tokens)[1]):
                        infos['candidates'][b]['metrics'] = _get_metrics(target, output.tokens[:, b])
                else:
                    infos['candidates'][0]['metrics'] = _get_metrics(target, output)
            
            return infos
        
        def final_filtering(infos, ** kwargs):
            """ Removes the `output` and `encoded` keys to save memory """
            infos.pop('output')
            infos.pop('encoded')
            return infos
        
        
        pred_config = self.training_hparams.extract(kwargs, pop = True)
        self.init_train_config(** pred_config)
        
        logger.dev('Prediction config :\n{}'.format(pred_config))
        
        if metrics is not None: metrics = self.get_compiled_metrics(metrics, add_loss = False)

        attn_dir = None
        if save:
            if directory is None: directory = self.pred_dir
            if map_file is None or directory.endswith('.json'): map_file, directory = directory, None
            else: map_file = os.path.join(directory, map_file)
            if directory is not None: os.makedirs(directory, exist_ok = True)
            
            if save_attention:
                attn_dir = os.path.join(os.path.dirname(map_file), 'attention')
                os.makedirs(attn_dir, exist_ok = True)

        expected_keys = ['candidates']
        do_not_save_keys    = ['encoded', 'output']
        
        final_functions = []
        if metrics is not None:
            final_functions.append({'consumer' : compute_metrics, 'allow_multithread' : False})
        
        if analyze_attention:
            final_functions.append(attention_analyzis)
        
        final_functions.append({'consumer' : final_filtering, 'max_workers' : -2})
        
        pipeline    = Pipeline(tasks = [
            {'consumer' : add_data_id, 'name' : 'pre_processing'},
            {
                'name'  : 'nlp_inference',
                'filename'  : map_file if save else None,
                'expected_keys' : expected_keys,
                'do_not_save_keys'  : do_not_save_keys,

                'tasks' : [
                    {'consumer' : encode, 'name' : 'pre_processing'},
                    {'consumer' : inference, 'batch_size' : batch_size, 'allow_multithread' : False},
                    decode
                ] +  final_functions
            }
        ], ** kwargs)
        pipeline.start()
        return pipeline
    
    def analyze_last_attention(self,
                               inputs,
                               attn_weights,
                               input_length   = None,
                               
                               k      = 10,
                               attn_name    = None,
                               reduction  = tf.reduce_sum,

                               skip_eos   = True,
                               
                               save   = False,
                               directory  = None,
                               filename   = None,
                               
                               return_indexes   = False,
                               return_attention  = False,
                               
                               ** kwargs
                              ):
        """
            Analyzes the model's last attention to give some information about it. It has been introduced in the `Memory Augmented Generator : a new approach for question-answering` from @Ananas120 (https://github.com/Ananas120/mag) and seems to be relevant to evaluate the confidence of an answer in Q&A.
            
            **Important Note** : this function has been tested for generator models (models that generates text), not yet for other types of models (such as span retriever). 
            In theory it should work but let me know if there are some issues ;)
            
            Arguments :
                - inputs    : the encoded model's inputs
                - attn_weights  : either the attention's weights to analyze, either the model's returned dict
                - input_length  : the inputs' lengths
                
                - k     : the top-k scores to analyze
                - attn_name : if `attn_weights` is a dict, uses `attn_weights[attn_name]` (default to the key with the highest index)
                - reduction : the function to apply to reduce the heads (default to the sum)
                - skip_eos  : whether to skip the EOS token's attention or not
                
                - save  : whether to save the attention's weights in a `.npy` file
                - filename / directory  : where to save the attention's weights
                
                - return_indexes    : whether to keep the `indexes / values / tokens` keys in the spans' information
                - return_attention  : whether to return the attention's weights in the infos or not
            Returns : (global_infos, attn_infos)
                - global_infos (dict)   : global information about the 1st attention's token (global to each candidate as the 1st output is identical)
                    - highest_attention_score   : 1st token's highest score
                    - highest_attention_span_score  : highest span score based on the 1st token's scores
                    - attention_shape   : the attention shape :D
                - attn_infos (dict)     : specific candidate's related information about the span's scores etc.
                    - filename  : where the attention is saved (None if not save)
                    - indexes / scores / para_indexes   : top-k scores with their indexes and paragraphs' indexes (if `self.multi_input_format`)
                    - spans : the list of tuple (span_text, span_score, span_infos) where each span_text is a sentence and span_score is the sum of individual token's scores for this span of text
            
            See the paper to better understand how to analyze these information and how they can potentially be used to analyze the model's confidence
        """
        tokens = inputs
        if isinstance(inputs, (list, tuple)):
            flattened, lengths = [], []
            if self.input_format is not None:
                inp, inp_len = inputs[:2]
                flattened.append(inp)
                lengths.append(inp_len)
            
            if self.input_multi_format is not None:
                multi_inp, multi_lengths = inputs[-2:]
                multi_inp = [m_inp[:l] for m_inp, l in zip(multi_inp, multi_lengths)]
                
                flattened += multi_inp
                lengths.extend(multi_lengths)

            tokens  = tf.concat(flattened, axis = -1)
            input_length    = tf.cast(lengths if len(lengths) > 1 else lengths[0], tf.int32)
            
            logger.debug('Flattened shape : {} - lengths : {}'.format(
                tuple(tokens.shape), input_length
            ))

        time_logger.start_timer('pre-processing')
        
        if attn_name is None:
            use_enc_attn = any(k.startswith('enc_attn') for k in attn_weights.keys())
            last_idx     = -1
            for key in attn_weights.keys():
                if use_enc_attn and not key.startswith('enc_attn'): continue
                k_idx = int(key.split('_')[-1])
                if k_idx > last_idx:
                    attn_name, last_idx = key, k_idx
        
        last_attn = attn_weights.get(attn_name, None)
        if last_attn is None:
            logger.warning('Attention {} not found in `attn_weights`\n  Accepted : {}'.format(
                attn_name, tuple(attn_weights.keys())
            ))
            time_logger.stop_timer('pre-processing')
            return {}
        
        filename = None
        if save:
            if directory is None: directory = os.path.join(self.pred_dir, 'attention')
            if filename is None:
                filename = 'attn_{}.npy'.format(len(glob.glob(
                    os.path.join(directory, 'attn_*.npy')
                )))
            filename = os.path.join(directory, filename)
            np.save(filename, np.array(last_attn))
        
        if len(tf.shape(last_attn)) == 4: last_attn = last_attn[0]
        if len(tf.shape(last_attn)) == 3:
            logger.debug('Reducing the attention\'s {} heads'.format(tf.shape(last_attn)[0]))
            last_attn = reduction(last_attn, axis = 0)
        
        if skip_eos: last_attn = last_attn[:-1]

        logger.debug('Analyzing attention with shape {}'.format(last_attn.shape))

        time_logger.stop_timer('pre-processing')
        time_logger.start_timer('Indices extraction')
        
        top_k = tf.nn.top_k(last_attn, k = k)
        indices, values = top_k.indices, top_k.values
        para_indices = None
        if input_length is not None and len(tf.shape(input_length)) > 0:
            if self.subsampling_factor > 1:
                if not self.subsample_input or self.input_format is None:
                    indices = indices * self.subsampling_factor
                else:
                    inp_len = input_length[0]
                    indices = inp_len + (indices - inp_len) * self.subsampling_factor
                indices = tf.clip_by_value(indices, 0, len(tokens) - 1)
            # compute the paragraphs' indices based on indices
            mask = tf.cast(tf.math.cumsum(input_length) < tf.reshape(indices, [-1, 1]), tf.int32)

            valids = tf.range(len(input_length)) * tf.expand_dims(mask, axis = 0)
            valids = valids + (1 - mask) * -1

            para_indices = tf.reshape(tf.reduce_max(valids, axis = -1), tf.shape(indices)).numpy()

        tokens, indices, values = tokens.numpy(), indices.numpy(), values.numpy()
        
        logger.debug('Top {} indexes / values for last attention :\n{}\n{}'.format(
            k, indices, values
        ))
        
        time_logger.stop_timer('Indices extraction')
        time_logger.start_timer('Post processing')

        total_attn_score_per_token   = tf.reduce_sum(last_attn, axis = -1).numpy()
        
        ranges, spans, idx_to_token = {}, {}, {}
        for i, total_attn in enumerate(total_attn_score_per_token):
            time_logger.start_timer('attention_i analysis')

            for idx, val in zip(indices[i], values[i]):
                time_logger.start_timer('index_i analysis')
                sent = None
                for (start, end), _sent in ranges.items():
                    if idx in range(start, end):
                        sent = _sent
                        break
                
                if sent is None:
                    time_logger.start_timer('sentence extraction')
                    
                    sent, start, end = self.text_encoder.extract_sentence(tokens, idx)
                    ranges[(start, end)] = sent
                    
                    time_logger.stop_timer('sentence extraction')

                _token_idx = tokens[idx]
                if _token_idx not in idx_to_token:
                    idx_to_token[_token_idx] = self.text_encoder.decode(
                        [_token_idx], remove_tokens = False
                    )
                
                spans.setdefault(sent, {
                    'score' : {}, 'normalized_score' : {}, 'indexes' : {}, 'values' : {}, 'tokens' : {}
                })
                spans[sent]['indexes'].setdefault(i, []).append(idx)
                spans[sent]['values'].setdefault(i, []).append(val)
                spans[sent]['tokens'].setdefault(i, []).append(idx_to_token[_token_idx])
                spans[sent]['score'].setdefault(i, 0.)
                spans[sent]['normalized_score'].setdefault(i, 0.)
                
                spans[sent]['score'][i] += val 
                spans[sent]['normalized_score'][i] += val / total_attn
                
                time_logger.stop_timer('index_i analysis')

            time_logger.stop_timer('attention_i analysis')
        
        span_infos = []
        for sent, infos in spans.items():
            if self.input_format is not None and self.input_multi_format is not None:
                inp_len = input_length[0]
                _indexes = []
                for _, v in infos['indexes'].items(): _indexes.extend(v)
                infos['is_question'] = all(idx <= inp_len for idx in _indexes)
            
            if not return_indexes:
                for k in ('indexes', 'values', 'tokens'): infos.pop(k)
            
            span_infos.append((
                sent, sum(infos['score'].values()) / len(last_attn), infos
            ))
        span_infos = sorted(span_infos, key = lambda i: i[1], reverse = True)
        
        #conf = span_infos[0][2]['score'].get(0, 0)
        conf    = max([
            infos['score'].get(0, 0) for _, _, infos in span_infos
        ])
        
        time_logger.stop_timer('Post processing')

        global_infos    = {
            'highest_attention_score'   : np.max(last_attn[0]),
            'highest_attention_span_score'  : conf,
            'attention_shape'   : tuple(last_attn.shape)
        }
        attn_infos      = {'spans' : span_infos}
        if filename is not None:
            attn_infos['filename'] = filename
        
        if return_indexes:
            attn_infos.update({
                'indexes'   : indices,
                'scores'    : values,
                'para_indexes'  : para_indices
            })
        
        if return_attention:
            attn_infos['attention'] = last_attn
        
        return global_infos, attn_infos

    @timer
    def predict(self, data, ** kwargs):
        """ Performs the inference pipeline on `data` """
        if not isinstance(data, (list, pd.DataFrame)): data = [data]
        pred_hparams   = self.training_hparams.extract(kwargs, pop = True)
        self.init_train_config(** pred_hparams)
        
        pipeline = self.get_pipeline(** kwargs)
        
        return pipeline.extend_and_wait(data, stop = True, ** kwargs)

    def get_config(self, * args, ** kwargs):
        config = super(BaseNLUModel, self).get_config(* args, ** kwargs)
        multi_input_config = {} if self.input_multi_format is None else {
            'input_multi_format'    : self.input_multi_format,
            'subsample_input'       : self.subsample_input,
            'multi_input_offset'    : self.multi_input_offset,
            
            'use_multi_input'   : self.use_multi_input,
            'split_multi_input' : self.split_multi_input,
            'split_key'         : self.split_key,
            'max_split_sentences'   : self.max_split_sentences,
            'max_sentence_length'   : self.max_sentence_length,
            'sort_by_length'    : self.sort_by_length,
            'max_total_length'  : self.max_total_length,
            'max_input_texts'   : self.max_input_texts,
            'input_select_mode' : self.input_select_mode
        }
        config.update({
            ** self.get_config_text(),
            'input_format'  : self.input_format,
            'output_format' : self.output_format,
            
            ** multi_input_config,
            
            'max_input_length'  : self.max_input_length,
            'use_fixed_length_input'    : self.use_fixed_length_input
        })
        return config
        
