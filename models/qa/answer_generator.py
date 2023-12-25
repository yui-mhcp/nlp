# Copyright (C) 2022-now yui-mhcp project's author. All rights reserved.
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

class AnswerGenerator(BaseNLUGenerator):
    def __init__(self,
                 * args,
                 input_format   = '{question}{sep_token}{context}',
                 output_format  = '{answer}',
                 ** kwargs
                ):
        super().__init__(
            * args, input_format = input_format, output_format = output_format, ** kwargs
        )
    
