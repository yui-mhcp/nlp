
# Copyright (C) 2022 yui-mhcp project's author. All rights reserved.
# Licenced under the Affero GPL v3 Licence (the "Licence").
# you may not use this file except in compliance with the License.
# See the "LICENCE" file at the root of the directory for the licence information.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" TF 2.0 OpenAI GPT-2 model, compatible with the `transformers`' checkpoint."""

import tensorflow as tf

from custom_architectures.transformers_arch.text_transformer_arch import TextTransformerEncoder, HParamsTextTransformerEncoder

HParamsBaseGPT2  = HParamsTextTransformerEncoder(
    use_causal_attention    = True,
    normalize_embeddings    = False,
    
    normalize   = 'middle',
    ffn_dim     = 3072,
    ffn_activation  = 'gelu_new',
    mha_normalize   = False,
    mha_normalize_input = True,
    mha_epsilon     = 1e-5,
    epsilon     = 1e-5
)

class BaseGPT2(TextTransformerEncoder):
    default_params  = HParamsBaseGPT2
    
    def __init__(self, vocab_size, embedding_dim, ** kwargs):
        super().__init__(vocab_size = vocab_size, embedding_dim = embedding_dim, ** kwargs)
        
        self.norm       = tf.keras.layers.LayerNormalization(
            epsilon = self.hparams.epsilon, name = 'norm_final'
        )

    def compute_output(self, output, training = False, mask = None, ** kwargs):
        return self.norm(output, training = training and self.norm_training)

    def transfer_weights(self, pretrained, tqdm = lambda x: x, ** kwargs):
        from models.weights_converter import _transformer_patterns, _attn_split, name_based_partial_transfer_learning

        return name_based_partial_transfer_learning(
            self, pretrained, patterns = _transformer_patterns,
            transforms = {** _attn_split, '.*' : lambda k, v: {k : [vi.T for vi in v]}}, tqdm = tqdm, ** kwargs
        )
        
    @classmethod
    def from_pretrained(cls,
                        pretrained_name = 'gpt2',
                        pretrained_task = 'generation',
                        pretrained      = None,
                        tqdm    = lambda x: x,
                        ** kwargs
                       ):
        from models.weights_converter import partial_transfer_learning

        if pretrained is None:
            from transformers import TFGPT2Model
            with tf.device('cpu') as d:
                pretrained = TFGPT2Model.from_pretrained(pretrained_name)

        if isinstance(pretrained, dict):
            pretrained  = {k : v for k, v in pretrained.items() if 'gpt' in k}
            n_layer     = len([k for k in pretrained if k.endswith('attn.weight')])
            
            config = HParamsBaseGPT2(
                vocab_size      = pretrained['gpt.transformer.wte.weight'].shape[0],
                embedding_dim   = pretrained['gpt.transformer.wte.weight'].shape[1],
                max_input_length    = pretrained['gpt.transformer.wpe.weight'].shape[0],
                sos_token   = 50256,
                eos_token   = 50256,

                num_layers  = n_layer,
                mha_num_heads   = 12
            )
        else:
            config = HParamsBaseGPT2(
                vocab_size      = pretrained.config.vocab_size,
                embedding_dim   = pretrained.config.n_embd,
                max_input_length    = pretrained.config.n_positions,
                sos_token   = 50256,
                eos_token   = 50256,

                num_layers  = pretrained.config.n_layer,
                mha_num_heads   = pretrained.config.n_head
            )

        instance = cls(** config(** kwargs))
        instance._build()

        instance.transfer_weights(pretrained, tqdm = tqdm, ** kwargs)

        return instance

class GPT2(BaseGPT2):
    @property
    def output_last_dim(self):
        return self.vocab_size
    
    def compute_output(self, output, training = False, mask = None, ** kwargs):
        output = super().compute_output(output, training = training, mask = mask, ** kwargs)
        
        return self.embeddings.linear(output)
    
custom_functions    = {
    'BaseGPT2'      : BaseGPT2,
    'GPT2'          : GPT2
}

custom_objects  = custom_functions

_encoders   = {'GPT2' : GPT2}
_transformers   = _encoders