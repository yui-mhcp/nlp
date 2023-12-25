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

import os
import glob
import logging
import numpy as np
import pandas as pd
import tensorflow as tf

from loggers import timer, time_logger
from utils.generic_utils import normalize_key
from models.interfaces.base_text_model import BaseTextModel
from custom_architectures.transformers_arch import get_pretrained_transformer
from custom_architectures.transformers_arch.mag_wrapper import MAGWrapper

logger      = logging.getLogger(__name__)

_alternative_keys   = {
    'text'      : ('text', 'context', 'content', 'paragraph'),
    'answer'    : ('answer', 'answers'),
    'context'   : ('context', 'paragraph', 'text', 'contexts', 'paragraphs', 'texts'),
    'title'     : ('title', 'titles'),
    'required_idx'  : ('valid_idx', )
}
_multi_alternative_keys   = {
    'answer'    : ('answers', 'answer'),
    'context'   : ('contexts', 'paragraphs', 'texts', 'context', 'paragraph', 'text'),
    'title'     : ('titles', 'title'),
    'required_idx'  : ('valid_idx', )
}


_non_trainable_config   = ('multi_input_format', 'multi_input_offset', 'split_key')

DEFAULT_MAX_INPUT_LENGTH    = 512

class BaseNLUModel(BaseTextModel):
    decode_output   = BaseTextModel.decode_text
    
    def __init__(self,
                 lang,
                 
                 input_format,
                 output_format,
                 
                 multi_input_format = None,     # format for multi contexts
                 multi_input_offset = -1,       # positional offset for multi contexts
                 
                 split_key          = None,     # the key to use to split multi_input
                 max_split_sentences    = -1,   # max number of sentences per context
                 max_sentence_length    = -1,   # maximum length for a sentence
                 
                 max_total_length   = -1,       # max total (cumulated) length for multi_input
                 sort_by_length     = False,    # sorting strategy for multi_input selection
                 
                 max_input_texts    = -1,       # maximum number of texts to keep in multi_input 
                 input_select_mode  = 'start',  # the keeping strategy to filter out contexts
                 
                 max_input_length   = DEFAULT_MAX_INPUT_LENGTH,
                 max_multi_input_length = -1,
                 
                 pretrained = None,
                 
                 ** kwargs
                ):
        self._init_text(lang = lang, ** kwargs)
        
        self.input_format   = input_format
        self.output_format  = output_format
        
        self.multi_input_format = kwargs.pop('input_multi_format', multi_input_format)
        self.multi_input_offset = multi_input_offset
        
        self.split_key  = split_key
        self.max_split_sentences    = max_split_sentences
        self.max_sentence_length    = max_sentence_length
        
        self.sort_by_length     = sort_by_length
        self.max_total_length   = max_total_length
        
        self.max_input_texts    = max_input_texts
        self.input_select_mode  = input_select_mode
        
        self.max_input_length   = max_input_length
        self.max_multi_input_length = max_multi_input_length
        
        self.merge_multi_input  = False
        
        if isinstance(pretrained, str): kwargs.setdefault('pretrained_name', pretrained)
        super(BaseNLUModel, self).__init__(pretrained = pretrained, ** kwargs)
        
        if hasattr(self.model, 'set_tokens'): self.model.set_tokens(** self.model_tokens)
    
    def _build_model(self, pretrained = None, ** kwargs):
        if pretrained is not None:
            super(BaseNLUModel, self)._build_model(
                model = get_pretrained_transformer(pretrained, ** kwargs)
            )
        else:
            super(BaseNLUModel, self)._build_model(** kwargs)

    @property
    def is_encoder_decoder(self):
        return False
    
    @property
    def encoder(self):
        return self.model if not self.is_encoder_decoder else getattr(self.model, 'encoder', None)
    
    @property
    def decoder(self):
        return getattr(self.model, 'decoder', None)

    @property
    def use_multi_input(self):
        return self.multi_input_format is not None
    
    @property
    def split_multi_input(self):
        return self.split_key is not None
    
    @property
    def subsampling_factor(self):
        return -1 if not self.use_multi_input else self.encoder.subsampling_step

    @property
    def subsample_multi_input(self):
        return self.use_multi_input and self.subsampling_factor > 1
    
    @property
    def input_signature(self):
        signature = ()
        if self.input_format is not None:   signature += (self.text_signature, )
        if self.use_multi_input:            signature += (self.multi_text_signature, )
        return signature if len(signature) > 1 else signature[0]

    @property
    def multi_input_config(self):
        return {} if self.use_multi_input else {
            'multi_input_format'    : self.multi_input_format,
            'multi_input_offset'    : self.multi_input_offset,
            
            ** self.split_multi_input_config,
            
            'sort_by_length'    : self.sort_by_length,
            'max_total_length'  : self.max_total_length,
            'max_input_texts'   : self.max_input_texts,
            'input_select_mode' : self.input_select_mode,
            'max_multi_input_length'    : self.max_multi_input_length
        }

    @property
    def split_multi_input_config(self):
        return {} if not self.split_multi_input else {
            'split_key'         : self.split_key,
            'max_split_sentences'   : self.max_split_sentences,
            'max_sentence_length'   : self.max_sentence_length,
        }
    
    @property
    def filter_multi_input_config(self):
        return {} if not self.use_multi_input else {
            'min_text_length'   : -1,
            'max_text_length'   : self.max_multi_input_length if self.max_multi_input_length != -1 else self.max_input_length,
            'max_sentences'     : self.max_split_sentences if self.split_multi_input else -1,
            'max_sentence_length'   : self.max_sentence_length if self.split_multi_input else -1,
            
            'max_total_length'  : self.max_total_length,
            'sort_by_length'    : self.sort_by_length,

            'max_texts'     : self.max_input_texts,
            'select_mode'   : self.input_select_mode
        }

    @property
    def training_hparams(self):
        multi_input_config = {
            k : None for k in self.multi_input_config if k not in _non_trainable_config
        }
        if multi_input_config:
            multi_input_config.update({
                'input_select_mode' : 'random',
                'merge_multi_input' : False
            })
        
        return super(BaseNLUModel, self).training_hparams(
            ** multi_input_config, max_input_length = None
        )
    
    @property
    def default_metrics_config(self):
        return {
            'pad_value' : self.blank_token_idx,
            'eos_value' : self.eos_token_idx,
            'decode_fn' : lambda text: self.decode_text(text, remove_tokens = True)
        }

    def __str__(self):
        des = super(BaseNLUModel, self).__str__()
        des += self._str_text()
        if self.input_format:
            des += "- Input format : {}\n".format(self.input_format)
        if self.multi_input_format:
            des += "- Multi input format : {}\n".format(self.multi_input_format)
        if self.output_format:
            des += "- Output format : {}\n".format(self.output_format)
        
        if self.use_multi_input:
            if self.split_multi_input:
                des += "- Split multi input key   : {}\n".format(self.split_key)
                des += "- Max sentences per split : {}\n".format(self.max_split_sentences)
                if self.self.max_sentence_length:
                    des += "- Max sentence lengths    : {}\n".format(self.max_sentence_length)
                
            des += "- # of memory layers    : {}\n".format(len(self.encoder.memory_layers))
            des += "- # of embedding layers : {}\n".format(len(self.encoder.embedding_layers))
            des += "- Subsampling factor : {}\n".format(self.subsampling_factor)
            des += "- Subsampling mode : {}\n".format(self.encoder.subsampling_mode)
        
        des += "- Max input length : {}\n".format(self.max_input_length)
        if self.multi_input_format and self.split_multi_input:
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
        if not self.use_multi_input: return {}
        # only compute statistics on inputs (and not output)
        if self.is_encoder_decoder and is_call: inputs = inputs[0]
        
        positional_offset = self.multi_input_offset
        if isinstance(inputs, (list, tuple)):
            positional_offset   = [positional_offset] * len(inputs)
            force_not_subsampling   = [force_not_subsampling] * len(inputs)
            
            if self.input_format is not None:
                positional_offset[0]    = -1
                force_not_subsampling[0]    = True
        
        merge_embeddings = merge_multi_input or (training and self.merge_multi_input)
        
        prefix = '' if not self.is_encoder_decoder else 'encoder_'
        return {
            '{}merge_embeddings'.format(prefix)         : merge_embeddings,
            '{}force_not_subsampling'.format(prefix)    : force_not_subsampling,
            '{}positional_offset'.format(prefix)        : positional_offset
        }

    @timer(name = 'prediction', log_if_root = False)
    def call(self, inputs, training = False, ** kwargs):
        kwargs.update(self._get_mag_config(inputs, True, training = training, ** kwargs))
        return self.model(inputs, training = training, ** kwargs)

    def infer(self, inputs, training = False, ** kwargs):
        kwargs.update(self._get_mag_config(inputs, False, training = training, ** kwargs))
        if hasattr(self.model, 'infer'):
            return self.model.infer(inputs, training = training, ** kwargs)
        return self(inputs, training = training, ** kwargs)
    
    def format_input(self, * args, ** kwargs):
        kwargs = {normalize_key(k, _alternative_keys) : v for k, v in kwargs.items()}
        return self.format_text(self.input_format, * args, ** kwargs)
    
    def format_multi_input(self, * args, flatten = True, ** kwargs):
        kwargs = {normalize_key(k, _multi_alternative_keys) : v for k, v in kwargs.items()}
        kwargs.update(self.filter_multi_input_config)
        if self.split_multi_input:
            kwargs.update({'split_key' : self.split_key, 'max_length' : self.max_sentence_length})
            if flatten:
                kwargs.update({
                    'flatten'   : True,
                    'shape'     : [tf.TensorShape((None, None)), tf.TensorShape((None, ))]
                })
            return self.multi_split_and_format_text(self.multi_input_format, * args, ** kwargs)
        
        return self.multi_format_text(self.multi_input_format, * args, ** kwargs)
    
    def format_output(self, * args, ** kwargs):
        kwargs = {normalize_key(k, _alternative_keys) : v for k, v in kwargs.items()}
        return self.format_text(self.output_format, * args, ** kwargs)
    
    def multi_format_output(self, * args, ** kwargs):
        kwargs = {normalize_key(k, _multi_alternative_keys) : v for k, v in kwargs.items()}
        return self.multi_format_text(self.output_format, * args, ** kwargs)
    
    def get_input(self, data = None, ** kwargs):
        if data is None: data = {}
        elif not isinstance(data, (dict, pd.Series)): data = {'text' : data}
        
        inputs = ()
        if self.input_format is not None:
            inputs += (self.format_input(** data, ** kwargs), )
        
        if self.use_multi_input:
            inputs += (self.format_multi_input(** data, ** kwargs)[0], )
        
        return inputs if len(inputs) > 1 else inputs[0]
    
    def get_output(self, data = None, inputs = None, ** kwargs):
        if self.output_format:
            if data is None: data = {}
            elif not isinstance(data, (dict, pd.Series)): data = {'text' : data}
            return self.format_output(** data, ** kwargs)
        raise NotImplementedError()
    
    def encode_data(self, data):
        """ Returns `(self.get_input(data), self.get_output(data))` """
        inputs  = self.get_input(data)
        outputs = self.get_output(data, inputs = inputs)

        return inputs, outputs
    
    def filter_input(self, inputs):
        """ Check `is_valid_tokens` for information """
        if self.is_encoder_decoder:         inputs = inputs[0]
        if not isinstance(inputs, tuple):   inputs = (inputs, )
        
        return tf.logical_and(
            tf.shape(inputs[0])[0] > 0, tf.shape(inputs[0])[-1] <= self.max_input_length
        )
    
    def filter_output(self, outputs):
        return True
    
    def filter_data(self, inputs, outputs):
        return self.filter_input(inputs) and self.filter_output(outputs)
    
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

    def get_config(self, * args, ** kwargs):
        config = super(BaseNLUModel, self).get_config(* args, ** kwargs)
        config.update({
            ** self.get_config_text(),
            'input_format'  : self.input_format,
            'output_format' : self.output_format,
            
            ** self.multi_input_config,
            
            'max_input_length'  : self.max_input_length
        })
        return config
        
