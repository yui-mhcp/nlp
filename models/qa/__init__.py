
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
from models.qa.rag import RAG
from models.qa.answer_retriever import AnswerRetriever
from models.qa.answer_generator import AnswerGenerator
from models.qa.question_generator import QuestionGenerator
from models.qa.context_retriever import ContextRetriever
from models.qa.text_encoder_decoder import TextEncoderDecoder
from models.qa.answer_generator_split import AnswerGeneratorSplit

from models.qa.web_utils import prepare_web_data, search_on_web

_default_pred_config    = {
    'method'            : 'beam',
    'negative_mode'     : 'doc',
    'max_negatives'     : 16,
    'negative_select_mode'  : 'linear',
    'max_input_length'  : 666,
    'min_context_length'    : 48,
    'max_total_length'  : 13000
}

def _get_model_name(model = None, lang = None):
    if model is not None: return model
    
    global _pretrained
    
    if lang not in _pretrained:
        raise ValueError("Unknown language : {}, no default model set".format(lang))
        
    return _pretrained[lang]

def answer(question, model = None, lang = None, ** kwargs):
    from models import get_pretrained
    
    model_name  = _get_model_name(model = model, lang = lang)
    model   = get_pretrained(model_name)
    
    pred_config  = {** _default_pred_config, ** kwargs}
    
    return model.predict(question, ** pred_config)

def answer_from_web(question, model = None, lang = 'en', ** kwargs):
    pred_config  = {** _default_pred_config, ** kwargs}
    
    data = prepare_web_data(question, ** pred_config)
    
    config = data['config'].copy()
    data['config']['model'] = _get_model_name(model = model, lang = lang)

    return answer([data], model = model, lang = lang, ** config)

_models = {
    'MAG'           : MAG,
    'RAG'           : RAG,
    'QARetriever'   : AnswerRetriever,
    'AnswerGenerator'   : AnswerGenerator,
    'QuestionGenerator' : QuestionGenerator,
    'ContextRetriever'  : ContextRetriever,
    'TextEncoderDecoder'    : TextEncoderDecoder,
    'AnswerGeneratorSplit'  : AnswerGeneratorSplit
}

_pretrained = {
    'en'    : 'm6_nq_coqa_newsqa_mag_off_entq_ct_wt_ib_2_2_dense'
}