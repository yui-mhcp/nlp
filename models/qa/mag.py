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

from models.nlu.base_nlu_generator import BaseNLUGenerator

class MAG(BaseNLUGenerator):
    def __init__(self,
                 * args,
                 
                 question_format    = None,
                 context_format     = None,
                 
                 context_offset     = -1,
                 split_contexts     = False,
                 subsample_question = True,
                 max_sentence_length    = 128,
                 
                 min_context_length = -1,
                 max_total_length   = -1,
                 sort_by_length     = False,
                 
                 skip_question_eos  = False,
                 skip_context_sos   = False,
                 
                 ** kwargs
                ):
        if question_format:
            kwargs.update({
                'input_format'  : question_format,
                'input_multi_format'    : context_format,
                'subsample_input'   : subsample_question,
                'multi_input_offset'    : context_offset,

                'use_multi_input'   : True,
                'split_multi_input' : split_contexts,
                'split_key'         : 'context',
                'max_sentence_length'   : max_sentence_length,
                'sort_by_length'    : sort_by_length,
                'max_total_length'  : max_total_length,
            })
        super().__init__(* args, ** kwargs)
