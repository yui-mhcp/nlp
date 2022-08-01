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
import logging
import numpy as np
import pandas as pd
import tensorflow as tf

from loggers import timer
from utils import load_json, dump_json, pad_batch
from utils.text import extract_sentence
from models.qa.base_qa import BaseQAModel
from models.interfaces.base_model import _compile_fn
from custom_architectures.transformers_arch.bart_arch import Bart
from custom_architectures.transformers_arch.gpt2_arch import GPT2

time_logger = logging.getLogger('timer')

_pred_classic_infos = [
    'question', 'context', 'title', 'paragraphs', 'titles', 'answers'
]

def infer_to_str(text, score, indent = 0):
    _indentation = ' ' * indent
    if not isinstance(text, (list, tuple)):
        return '{}Inference ({:.3f}) : {}'.format(_indentation, score, text)
    des = '{}Inference :'.format(_indentation)
    for j, (s, txt) in enumerate(zip(score, text)):
        des += '\n{}  #{} ({:.3f}) : {}'.format(_indentation, j, s, txt)
    return des

class BaseGenerator(BaseQAModel):
    def __init__(self,
                 * args,
                 max_output_length  = 1024,
                 pretrained         = 'facebook/bart-large',
                 ** kwargs
                ):
        self.max_output_length = max_output_length
        
        self.show_input = kwargs.get('show_input', True)
        super().__init__(* args, pretrained = pretrained, ** kwargs)
    
    def _build_model(self, pretrained, ** kwargs):
        kwargs.update({'return_attention' : False, 'return_hidden_states' : False})
        if 'bart' in pretrained:
            model = Bart.from_pretrained(pretrained, ** kwargs)
        elif 'gpt' in pretrained:
            model = GPT2.from_pretrained(pretrained, ** kwargs)
        super()._build_model(model = model)
    
    @property
    def training_hparams(self):
        return super().training_hparams(
            max_output_length = None, teacher_forcing_eval = True, eval_infer_config = {}
        )

    @property
    def is_encoder_decoder(self):
        return hasattr(self.model, 'decoder')
    
    @property
    def token_length_signature(self):
        return (
            tf.TensorSpec(shape = (None, None), dtype = tf.int32),  # tokens
            tf.TensorSpec(shape = (None, ),     dtype = tf.int32)   # length
        )
    
    @property
    def multi_token_length_signature(self):
        return (
            tf.TensorSpec(shape = (None, None, None),   dtype = tf.int32),  # tokens
            tf.TensorSpec(shape = (None, None),         dtype = tf.int32)   # length
        )

    @property
    def input_signature(self):
        signature = self.token_length_signature
        if self.is_encoder_decoder: signature = signature + signature
        return signature
    
    @property
    def output_signature(self):
        signature = self.multi_token_length_signature #token_length_signature
        if self.is_encoder_decoder:
            return signature
        return signature + (tf.TensorSpec(shape = (None, ), dtype = tf.int32), )
    
    @property
    def default_metrics_config(self):
        return {
            'pad_value' : self.blank_token_idx,
            'eos_value' : self.eos_token_idx,
            'decode_fn' : lambda text: self.decode_text(text, remove_tokens = True)
        }

    @timer(name = 'inference', log_if_root = False)
    def infer(self, text, text_length = None, ** kwargs):
        if isinstance(text, (list, tuple)): text, text_length = text
        if len(tf.shape(text)) == 1: text = tf.expand_dims(text, axis = 0)
        if text_length is None: text_length = tf.fill([tf.shape(text)[0]], tf.shape(text)[1])
        elif len(tf.shape(text_length)) == 0: text_length = tf.expand_dims(text_length, axis = 0)
        
        kwargs.setdefault('max_length', self.max_output_length)
        return self.model.infer([text, text_length], ** kwargs)
    
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
    
    def encode_multi_answers(self, answers, ** kwargs):
        if isinstance(answers, tf.Tensor): answers = answers.numpy()
        if not isinstance(answers, (list, tuple, np.ndarray)): answers = [answers]
        
        encoded = [
            self.format_output(answer = a if not isinstance(a, bytes) else a.decode('utf-8'))[0]
            for a in answers
        ]
        
        return pad_batch(encoded, pad_value = self.blank_token_idx), [len(a) for a in encoded]

    def tf_format_multi_output(self, data):
        answers = data.get('answers', ['']) if isinstance(data, dict) else data
        
        encoded_outputs, lengths    = tf.py_function(
            self.encode_multi_answers, [answers], Tout = [tf.int32, tf.int32]
        )
        encoded_outputs.set_shape([None, None])
        lengths.set_shape([None])
        
        valid_outputs   = lengths <= self.max_output_length

        encoded_outputs = tf.boolean_mask(encoded_outputs, valid_outputs)
        lengths         = tf.boolean_mask(lengths, valid_outputs)
        
        if len(lengths) > 0:
            encoded_outputs = encoded_outputs[:, : tf.reduce_max(lengths)]
        
        return encoded_outputs, lengths

    def get_output(self, data, inputs = None):
        tokens, lengths = self.tf_format_multi_output(data)
        
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
    
    def filter_inputs(self, inputs):
        return tf.shape(inputs[0])[-1] <= self.max_input_length
        
    def filter_outputs(self, outputs):
        return len(outputs[0]) > 0 and tf.shape(outputs[0])[-1] <= self.max_output_length
    
    def filter_data(self, inputs, outputs):
        return self.filter_inputs(inputs) and self.filter_outputs(outputs)
    
    def augment_data(self, inputs, outputs):
        inp_tokens, inp_length = inputs[:2]
        
        inp_tokens, inp_length = self.augment_text(inp_tokens, inp_length)
        
        return (inp_tokens, inp_length) + inputs[2:], outputs
    
    def preprocess_data(self, inputs, outputs):
        if self.is_encoder_decoder:
            answer, answer_length = outputs
            answer_in, answer_in_length = answer[..., :-1], answer_length -1
            
            if len(tf.shape(answer_in)) == 3:
                answer_in, answer_in_length = answer_in[:, 0], answer_in_length[:, 0]
                answer_in = answer_in[:, : tf.reduce_max(answer_in_length)]
            
            return inputs + (answer_in, answer_in_length), (answer[..., 1:], answer_length - 1)
        
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
        answers, answers_length = outputs[:2]
        infer_inputs    = inputs[:-2] if self.is_encoder_decoder else inputs
        
        pred    = self(inputs, training = False, ** kwargs)
        infer   = self.infer(
            infer_inputs, max_length = tf.shape(answers)[-1], early_stopping = False,
            ** self.eval_infer_config, ** kwargs
        )
        
        pred_text   = self.decode_text(pred)
        infer_text  = self.decode_text(infer.tokens)
        
        input_text  = self.decode_text(inputs[0]) if self.show_input else None
        target_text = self.decode_text(answers)
        
        preds = []
        for i in range(len(target_text)):
            preds.append("Prediction {} / {} :\n{}  Target     : {}\n  Prediction : {}\n{}".format(
                i + 1, len(target_text),
                "" if input_text is None else "  Input      : {}\n".format(input_text[i]),
                target_text[i],
                pred_text[i],
                '' if infer_text is None else infer_to_str(infer_text[i], infer.score[i], indent = 2)
            ))
        
        logging.info("\n".join(preds))
    
    @timer(name = 'attention analysis')
    def extract_spans(self,
                      tokens,
                      attn_weights,
                      k      = 10,
                      input_length   = None,
                      q_len      = None,
                      skip_eos   = True,
                      reduction  = tf.reduce_sum,
                      save   = False,
                      directory  = None,
                      filename   = None,
                      return_attention  = False,
                      ** kwargs
                     ):
        time_logger.start_timer('pre-processing')
        
        last_idx  = len(self.model.decoder) if self.is_encoder_decoder else len(self.model)
        last_attn = attn_weights.get(
            'enc_attn_layer_{}'.format(last_idx),
            attn_weights.get('attn_layer_{}'.format(last_idx), None)
        )
        if last_attn is None: return {}
        
        filename = None
        if save:
            if directory is None: directory = self.pred_dir
            if filename is None:
                filename = 'attn_{}.npy'.format(len([
                    f for f in os.listdir(directory) if f.startswith('attn_') and f.endswith('.npy')
                ]))
            os.makedirs(directory, exist_ok = True)
            filename = os.path.join(directory, filename)
            np.save(filename, np.array(last_attn))
        
        if skip_eos: last_attn = last_attn[:-1]
        
        if len(tf.shape(last_attn)) == 3:
            last_attn = reduction(last_attn, axis = 0)
        
        logging.debug('Analyzing attention with shape {}'.format(last_attn.shape))

        time_logger.stop_timer('pre-processing')
        time_logger.start_timer('indices analysis')
        
        top = tf.nn.top_k(last_attn, k = k)
        indices, values = top.indices, top.values
        para_indices = None
        if input_length is not None and len(tf.shape(input_length)) > 0:
            if self.subsampling_factor > 1:
                q_len = input_length[0]
                indices = tf.clip_by_value(
                    q_len + (indices - q_len) * self.subsampling_factor, 0, len(tokens) - 1
                )
            
            mask = tf.cast(tf.math.cumsum(input_length) < tf.reshape(indices, [-1, 1]), tf.int32)

            valids = tf.range(len(input_length)) * tf.expand_dims(mask, axis = 0)
            valids = valids + (1 - mask) * - 1

            para_indices = tf.reshape(tf.reduce_max(valids, axis = -1), tf.shape(indices)).numpy()

        tokens, indices, values = tokens.numpy(), indices.numpy(), values.numpy()
        
        time_logger.stop_timer('indices analysis')

        logging.debug('Top {} indexes / values for last attention :\n{}\n{}'.format(
            k, indices, values
        ))

        time_logger.start_timer('post-processing')

        total_attn_score_per_token   = tf.reduce_sum(last_attn, axis = -1).numpy()
        
        ranges, spans, idx_to_token = {}, {}, {}
        for i, total_attn in enumerate(total_attn_score_per_token):
            logging.debug('\n\nAttention index {}\n'.format(i))

            time_logger.start_timer('attention_i analysis')

            for idx, val in zip(indices[i], values[i]):
                time_logger.start_timer('index_i analysis')
                sent = None
                for (start, end), _sent in ranges.items():
                    if idx >= start and idx < end:
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
            if q_len is not None:
                _indexes = []
                for _, v in infos['indexes'].items(): _indexes.extend(v)
                infos['is_question'] = all([idx <= q_len for idx in _indexes])
            span_infos.append((
                sent, sum(infos['score'].values()) / len(last_attn), infos
            ))
        span_infos = sorted(span_infos, key = lambda i: i[1], reverse = True)
        
        #conf = span_infos[0][2]['score'].get(0, 0)
        conf    = max([
            infos['score'].get(0, 0) for _, _, infos in span_infos
        ])
        
        time_logger.stop_timer('post-processing')

        global_infos    = {
            'highest_attention_score'   : np.max(last_attn[0]),
            'highest_attention_span_score'  : conf,
            'attention_shape'   : tuple(last_attn.shape)
        }
        return global_infos, {
            'filename'  : filename,
            'indexes'   : indices,
            'scores'    : values,
            'para_indexes'  : para_indices,
            'spans'     : span_infos,
            'attention' : None if not return_attention else last_attn
        }

    @timer
    def predict(self,
                question,
                
                title   = None,
                context = None,
                
                metrics = None,
                
                save        = False,
                overwrite   = False,
                directory   = None,
                filename    = 'map.json',
                
                analyse_attention   = True,
                
                tqdm    = lambda x: x,
                
                ** kwargs
               ):
        time_logger.start_timer('processing')

        pred_config = self.training_hparams.extract(kwargs, pop = True)
        self.init_train_config(** pred_config)
        
        logging.dev('Predicting config :\n{}'.format(pred_config))

        if not hasattr(self, '_compiled_infer'):
            self._compiled_infer    = _compile_fn(
                self.infer,
                run_eagerly = kwargs.pop('run_eagerly', False)
            )
        
        if metrics is not None: metrics = self.get_compiled_metrics(metrics, add_loss = False)

        if isinstance(question, pd.DataFrame): question = question.to_dict('record')
        if not isinstance(question, list): question = [question]

        if context is not None:
            if not isinstance(context, list) or len(context) != len(question): context = [context]
            if len(context) == 1 and len(question) > 1: context = context * len(question)

            if title is not None:
                if not isinstance(title, list) or len(title) != len(context): title = [title]
                if len(title) == 1 and len(context) > 1: title = title * len(context)
        
        
        data = question if context is None else []
        if context is not None:
            for i, q in enumerate(question):
                if not isinstance(q, dict): q = {'question' : q}
                ctx = context[i] if len(context) == len(question) else context
                
                if not isinstance(ctx, dict):
                    key = 'paragraphs' if isinstance(ctx, (list, tuple)) else 'context'
                    ctx = {key : ctx}
                    if title is not None:
                        key = 'titles' if key == 'paragraphs' else 'title'
                        ctx[key] = title[i] if len(title) == len(question) else title
                
                data.append({** q, ** ctx})

        time_logger.stop_timer('processing')

        infos_pred = {}
        if save:
            if directory is None: directory = self.pred_dir
            if filename is None or '.json' in directory: filename, directory = directory, None
            else: filename = os.path.join(directory, filename)
            if directory is not None: os.makedirs(directory, exist_ok = True)

            infos_pred = load_json(filename)

        answers = []
        for idx, row in enumerate(tqdm(data)):
            question    = row['question']
            context     = row['context'] if 'paragraphs' not in row else row['paragraphs']
            
            ctx_key     = context if not isinstance(context, list) else '\n\n'.join(context)

            if overwrite or not (question in infos_pred and ctx_key in infos_pred[question]):
                inputs = [tf.expand_dims(inp, axis = 0) for inp in self.get_input(row)]

                if not self.filter_inputs([inp[0] for inp in inputs]):
                    logging.warning('Too long data at index {} : {}'.format(
                        idx, [tuple(inp.shape) for inp in inputs]
                    ))
                    continue

                used_paragraphs = self.decode_text(inputs[2][0])

                additional_infos    = {
                    k : v for k, v in row.items() if k not in _pred_classic_infos
                }
                additional_infos['paragraphs']    = used_paragraphs
                additional_infos['paragraphs_len']  = inputs[3][0].numpy()

                pred = self._compiled_infer(
                    inputs, training = False, return_last_attention = analyse_attention, ** kwargs
                )

                lengths     = pred.lengths[0].numpy()
                scores      = pred.score[0].numpy()
                pred_text   = self.decode_text(pred.tokens, remove_tokens = True)[0]
                if not isinstance(pred_text, (list, tuple)):
                    pred_text, scores = [pred_text], [scores]

                
                infos_pred.setdefault(question, {})
                infos_pred[question][ctx_key] = {
                    'question' : question, ** additional_infos, 'candidates' : []
                }
                
                target = []
                if 'answers' in row and metrics is not None:
                    infos_pred[question][ctx_key]['target'] = row['answers']
                    target = [
                        tf.expand_dims(out, axis = 0) for out in self.get_output(row)
                        if len(out) > 0
                    ]

                for i, (txt, s, txt_len) in enumerate(zip(pred_text, scores, lengths)):
                    metrics_i = {}
                    if len(target) > 0:
                        time_logger.start_timer('metrics')
                        
                        metrics.reset_states()
                        metrics.update_state(target, pred.tokens[:, i])
                        metrics_i = {
                            name : val for name, val in zip(metrics.metric_names, metrics.result().numpy())
                        }
                        metrics_i['answers_len'] = target[1][0].numpy()
                        
                        time_logger.stop_timer('metrics')
                    
                    passages, attn_infos    = [], None
                    if analyse_attention:
                        # attention has shape [batch_size, beams, n_heads, a_len, inp_len]
                        global_attn_infos, attn_infos = self.extract_spans(inputs, {
                            k : v[0, i, :, :txt_len] for k, v in pred.attention_weights.items()
                        }, save = save, directory = directory, ** kwargs)
                        
                        for k, v in global_attn_infos.items():
                            infos_pred[question][ctx_key].setdefault(k, v)
                        
                        passages = [span[0] for span in attn_infos['spans']]
                    else:
                        if '.' in txt:
                            if not isinstance(context, (list, tuple)): context = [context]
                            passages    = [c for c in additional_infos['paragraphs'] if txt in c]
                        else:
                            passages    = extract_sentence('\n\n'.join(used_paragraphs), txt)
                    
                    infos_i = {
                        'text'  : txt,
                        'score' : s,
                        'text_len'  : txt_len,
                        'passages'  : passages,
                        'attention_infos'   : attn_infos,
                        ** metrics_i
                    }
                    infos_pred[question][ctx_key]['candidates'].append(infos_i)

            answers.append(infos_pred[question][ctx_key])

        if save:
            dump_json(filename, infos_pred, indent = 4)

        return answers
        
    def get_config(self, * args, ** kwargs):
        config = super().get_config(* args, ** kwargs)
        config['max_output_length'] = self.max_output_length
        return config