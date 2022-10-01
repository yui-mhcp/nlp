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

import re
import tensorflow as tf

DEFAULT_MAX_INPUT_LENGTH    = 512
DEFAULT_MAX_OUTPUT_LENGTH   = 1024

key_pattern  = re.compile('\{\w+\}')

def _get_expected_keys(text_format):
    """
        Returns the expected keys from the given `text_format`
        Keys are defined as the names between "{}" in a `str.format` call
    """
    if not text_format: return None
    if not isinstance(text_format, (list, tuple)): text_format = [text_format]
    
    keys = []
    for sub_format in text_format:
        for match in re.findall(key_pattern, sub_format):
            key = match[1:-1]
            if not key.endswith('_token') and key not in keys: keys.append(key)
    return tuple(keys)

def _get_key_mapping(keys, alternatives):
    """
        Returns a mapping `{key : alternatives}` for all keys in `keys`
        
        Arguments :
            - keys  : list of str, the expected keys
            - alternatives  : list of list / tuple where each tuple contains multiple keys that should be matched to the same key
        Returns :
            - mapping   : a dict where the keys are the keys in `keys` and values is the tuple containing the `key`
    """
    if not keys: return None
    
    mapping = {}
    for k in keys:
        mapping.setdefault(k, [])
        for alt in alternatives:
            if k in alt: mapping[k].extend(alt)
    return mapping

def infer_to_str(text, score, indent = 0):
    _indentation = ' ' * indent
    if not isinstance(text, (list, tuple)):
        return '{}Inference ({:.3f}) : {}'.format(_indentation, score, text)
    
    des = '{}Inference :'.format(_indentation)
    for j, (s, txt) in enumerate(zip(score, text)):
        des += '\n{}  #{} ({:.3f}) : {}'.format(_indentation, j, s, txt)
    return des

def is_valid_tokens(tokens, max_length = -1):
    """
        Returns True if the 1st dimension of `tokens` is higher than 0 and its last dimension is smaller or equal than `max_length` (if positive)
    """
    return tf.logical_and(
        tf.shape(tokens)[0] > 0,
        tf.shape(tokens)[-1] <= max_length if max_length > 0 else True
    )

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