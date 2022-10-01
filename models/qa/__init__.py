
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

import logging
import requests

from models.qa.mag import MAG
from models.qa.answer_retriever import AnswerRetriever
from models.qa.answer_generator import AnswerGenerator
from models.qa.question_generator import QuestionGenerator

from models.qa.web_utils import prepare_web_data, search_on_web

_default_pred_config    = {
    'method'            : 'beam',
    'max_input_texts'   : 16,
    'input_select_mode' : 'start',
    'max_multi_input_length'    : 666,
    'min_multi_input_length'    : 48,
    'max_total_length'  : 13000
}

def get_qa_model(model = None, lang = None):
    if model is None:
        global _pretrained

        if lang not in _pretrained:
            raise ValueError("No default model for {}\nAccepted :\n{}".format(
                lang, '\n'.join('- {} : {}'.format(k, v) for k, v in _pretrained.items())
            ))
        
        model = _pretrained[lang]
    
    if isinstance(model, str):
        from models import get_pretrained
        model = get_pretrained(model)
    
    return model

def answer(question, model = None, lang = None, ** kwargs):
    model = get_qa_model(model = model, lang = lang)
    
    pred_config  = {** _default_pred_config, ** kwargs}
    
    return model.predict(question, ** pred_config)

def answer_from_web(question, model = None, lang = 'en', ** kwargs):
    model = get_qa_model(model = model, lang = lang)
    
    pred_config  = {** _default_pred_config, ** kwargs}
    
    data = prepare_web_data(question, ** pred_config)
    
    config = data['config'].copy()
    data['config']['model'] = model.nom

    return answer([data], model = model, lang = lang, ** config)

_models = {
    'MAG'       : MAG,
    'AnswerRetriever'   : AnswerRetriever,
    'AnswerGenerator'   : AnswerGenerator,
    'QuestionGenerator' : QuestionGenerator
}

_pretrained = {
    'en'    : 'maggie'
}