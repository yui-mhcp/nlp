
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

"""
This script automates experiments based on model's names : the name should contain all the information to build the model (determine its configuration), train and test it. 
It allows to automate multiple training in a single command simply by passing multiple names. 

Example for a BART Q&A fine-tuning experiment : 
```
python3 main.py experiments --test --test_name test_nq_top5 --pred --pred_name pred_squad --names bart_nq bart_nq_coqa_newsqa
```

In this case, you can observe : 
    --test  : tells to perform model's testing
    --test_name test_nq_top5    : gives a name to the test and `nq` can represent the Natural Questions dataset while `top5` can represent the usage of Beam Search decoding
        Note that these information should be returned by the `testing_config_from_name()` method
    
    --pred  : tells to perform prediction
    --pred_name pred_squad  : gives a name to the prediction and `squad` can represent the SQUAD dataset
        These information should be returned by the `predict_config_frop_name` method
    
    --names : gives the model's names to build, train and test
    - bart_nq / bart_nq_coqa_newsqa : the model's names. The 1st one should be trained on NQ while the second one should be trained on [NQ, CoQA, newsQA] datasets
        The datasets' information should be returned in the `training_config_from_name` method
        The `bart_` prefix can tells to use a BART-based model (`AnswerGenerator` class) which should be returned in the `config_from_name` method
"""

import os
import glob
import json
import logging
import subprocess
import tensorflow as tf

from utils import parse_args
from models.model_utils import _pretrained_models_folder, get_model_history, get_models, is_model_name

logger  = logging.getLogger(__name__)

PRED_DIR    = os.path.join('__predictions')

ALLOWED_DATASETS    = ('nq', 'squad', 'coqa', 'newsqa', 'french_squad', 'piaf', 'fquad')

def _is_barthez(model_name):
    return True if ('fmag' in model_name or 'fbart' in model_name) and '_big_' not in model_name else False

def _is_french_model(model_name):
    return True if 'fmag' in model_name or 'fbart' in model_name else False

def simple_generator_config(model_name, bart_base = None, ** kwargs):
    if bart_base is None:
        if '_big_' in model_name:
            bart_base = 'moussaKam/mbarthez'
        elif _is_barthez(model_name):
            bart_base = 'moussaKam/barthez'
        else:
            bart_base = 'facebook/bart-large'
    
    return {
        'class'     : 'AnswerGenerator',
        'nom'       : model_name,
        'lang'      : 'en' if not _is_french_model(model_name) else 'fr',
        'pretrained'    : bart_base,
        'text_encoder'  : bart_base,
        
        'input_format'      : ['{question}', '{context}'],
        'output_format'     : '{answer}',
        'text_encoder'      : bart_base,
        'max_input_length'  : 512,
        'max_output_length' : 128,
        ** kwargs
    }

def config_from_name(model_name, bart_base = None, ** kwargs):
    config = simple_generator_config(model_name, bart_base, ** kwargs)
    if 'mag' not in model_name: return config
    
    step, idx, mode = model_name.split('_')[-3 :]
    step, idx = int(step), int(idx)
    
    offset = -1
    if 'off64' in model_name:
        offset = 64
    elif 'off' in model_name:
        offset = 128
    
    max_neg = 32
    if '_split_' in model_name: max_neg = max_neg * 4
    
    config.update({
        'class'     : 'MAG',
        'input_format'  : '{question}',
        'subsample_input'   : True if 'splitq' in model_name else False,
        
        'split_key' : 'context',
        'input_multi_format' : '{context}' if 'ct' not in model_name else ['{title}', '{context}'],
        'multi_input_offset'    : offset,
        'split_multi_input' : True if '_split_' in model_name else False,
        
        'encoder_repeat_pos_idx'    : True if 'rep' in model_name else False,
        'encoder_subsample_at'      : idx,
        'encoder_subsample_after'   : True if idx == 12 else False,
        'encoder_subsampling_step'  : step,
        'encoder_subsampling_offset': 0,
        'encoder_subsampling_mode'  : mode,
        
        'encoder_use_type_embedding': True if 'wt' in model_name else False,
        'encoder_max_types'         : max_neg * step
    })
    
    if 'ft_doc' in model_name:
        config['pretrained_name'] = model_name.replace('ft_doc', 'ib')
    elif 'dense' in model_name:
        config['pretrained_name'] = model_name.replace('dense', 'mean')
    
    return config

def simple_train_generator_config(model_name, retraining = False, lr = 1e-5, ** kwargs):
    use_doc = 'mag' in model_name and ('doc' in model_name or 'qangaroo' in model_name)
    
    datasets    = [
        ds_name for ds_name in ALLOWED_DATASETS if ds_name in model_name
    ]
    if 'french_squad' in datasets: datasets.remove('squad')
    
    return {
        'dataset'   : datasets,
        
        'compile_config'    : {
            'optimizer' : 'adam', 'optimizer_config' : {'lr' : lr}
        },

        'dataset_config'    : {
            'allow_la'          : False if 'osa' in model_name else True,
            'clean_text'        : True,
            'keep_mode'         : 'longest',
            'skip_impossible'   : True,
            'include_document'  : use_doc,
            'shuffle'   : True
        },
        'is_rectangular'    : False if use_doc else True,

        'epochs'    : 1,
        'batch_size'    : 6 if not _is_barthez(model_name) else 12,
        
        'shuffle_size'  : 6 * 32,

        'max_input_length'  : 512,
        'max_multi_input_length'    : 512,
        'max_output_length' : 32 * 3,
        ** kwargs
    }

def training_config_from_name(model_name, retraining = False, ** kwargs):
    lr = 1e-5
    if 'dense' in model_name:
        lr = {'name' : 'DivideByStep', 'maxval' : 1e-5, 'minval' : 1e-6, 'factor' : 0.1}
    elif _is_french_model(model_name):
        lr = {'name' : 'DivideByStep', 'maxval' : 5e-5, 'minval' : 1e-5, 'factor' : 0.1}
        
    config = simple_train_generator_config(model_name, retraining, lr = lr, ** kwargs)
    if 'mag' not in model_name: return config
    
    use_doc = config['dataset_config']['include_document']

    step, idx, mode = model_name.split('_')[-3 :]
    step = int(step)
    
    if 'dense' in model_name or retraining:
        epochs = 1 if not retraining else int(retraining)
    else:
        epochs = max(1, step // 2 + 1)
        if '_split_' in model_name: epochs = max(2, epochs)
        if _is_barthez(model_name): epochs += 1
        if '_big_' in model_name: epochs += 1

    if step < 2:
        batch_size = 3
    elif step == 2:
        batch_size = 4
    elif step == 3:
        batch_size = 5
    elif step > 3:
        batch_size = 6

    if use_doc: batch_size = max(1, batch_size // 2)
    if '_split_' in model_name: batch_size = max(1, batch_size // 2)
    
    if _is_barthez(model_name) and '_split_' in model_name: batch_size = batch_size * 3
    if '_big_' in model_name and 'french_squad' in model_name: batch_size = max(1, batch_size - 1)
    
    max_texts = 4 if '_split_' not in model_name else 3
    
    config.update({
        'epochs'    : epochs,
        'batch_size'    : batch_size,
        'shuffle_size'  : batch_size * 32,
        
        'use_multi_input'   : use_doc,
        'merge_multi_input' : True if 'ib' in model_name else False,
        'max_input_texts'   : max_texts if use_doc or '_split_' in model_name else -1,
        'max_split_sentences'   : 5,
        'max_sentence_length'   : 128
    })
    
    return config

def testing_config_from_name(model_name, test_name, ** kwargs):
    if 'mag' not in model_name:
        step, idx, mode = 1, -1, None
    else:
        step, idx, mode = model_name.split('_')[-3 :]
        step, idx = int(step), int(idx)

    datasets = [
        ds_name for ds_name in ALLOWED_DATASETS if ds_name in test_name
    ]
    if len(datasets) == 0: datasets.append('nq')
    if 'french_squad' in datasets: datasets.remove('squad')

    use_doc = True if 'doc' in test_name else False

    mode = 'none'
    if use_doc: mode = 'doc'
    elif 'ib' in test_name: mode = 'batch'
        
    batch_size = 12
    if use_doc and 'top5' in test_name: batch_size = 1
    elif use_doc: batch_size = 3
    elif 'top5' in test_name: batch_size = 6
    
    config = {
        'dataset'   : datasets,
        'test_name' : test_name,

        'dataset_config'    : {
            'keep_mode'         : 'all' if 'all' in test_name else 'longest',
            'allow_la'          : False if 'osa' in test_name else True,
            'clean_text'        : True,
            'skip_impossible'   : True,
            'include_document'  : use_doc,
            'shuffle'   : True
        },
        'is_rectangular'    : False if use_doc or 'all' in test_name else True,
        
        'metrics'       : ['F1'] if 'top5' not in test_name else ['TopKF1'],
        'add_loss'      : False,
        'batch_size'    : batch_size,
        
        'teacher_forcing_eval'  : True if 'tf' in test_name else False,
        'eval_infer_config'     : {} if 'top5' not in test_name else {'method' : 'beam'},
        
        'max_input_length'      : 512,
        'max_multi_input_length'    : 512,
        'max_output_length'     : 32 * 3,
        
        #'run_eagerly'   : True,
        ** kwargs
    }
    
    if 'mag' in model_name:
        config.update({
            'input_select_mode' : 'start',
            'max_input_texts'   : (5 * step - 1) if not 'split' in model_name else 5,
            'max_split_sentences'   : 5,
            'max_sentence_length'   : 128,
        
            'use_multi_input'   : use_doc,
            'merge_multi_input' : True if 'ib' in test_name else False
        })
    
    return config

def predict_config_from_name(model_name, pred_name, ** kwargs):
    config = testing_config_from_name(
        model_name,
        pred_name,
        save        = True,
        directory   = os.path.join(PRED_DIR, model_name),
        filename    = '{}.json'.format(pred_name),
        ** kwargs
    )
    config.update({
        'is_rectangular'    : False,
        'metrics'       : ['f1'],
        'method'        : 'beam',
        'max_input_texts'   : 10
    })
    for k in ['add_loss', 'batch_size']: config.pop(k, None)
    
    config['dataset_config']['keep_mode'] = 'all'
    
    return config

def _run_command(mode, * args, ** config):
    config = config_to_list(config)
    
    call_args   = ['python3', 'main.py', mode] + list(args) + config
    logger.info('Call arguments : `{}`'.format(' '.join(call_args)))
    
    return subprocess.run(call_args)

def config_to_list(config):
    config_list = []
    for k, v in config.items():
        config_list.append('--{}'.format(k))
        if not isinstance(v, (list, tuple)): v = [v]
        config_list.extend([json.dumps(vi) if not isinstance(vi, str) else vi for vi in v])
    
    return config_list

def run_experiments(names = [], ** kwargs):
    logger.info('tensorflow version : {}\n# GPU : {}'.format(
        tf.__version__, len(tf.config.list_physical_devices('GPU'))
    ))
    tf.config.set_visible_devices([], 'GPU')
    
    default_config = parse_args('mode', add_unknown = True)
    default_config.pop('mode')

    pred        = default_config.pop('pred', False)
    pred_name   = None if not pred else default_config.pop('pred_name', 'pred')

    testing     = default_config.pop('test', False)
    test_name   = None if not testing else default_config.pop('test_name', 'test')
    
    overwrite   = default_config.pop('overwrite', False)
    
    names       = default_config.pop('names', names)
    allow_retraining    = default_config.pop('retrain', False)
    if not isinstance(names, (list, tuple)):
        names = get_models(names) if '*' in names else [names]
    
    logger.info('Pred : {} ({}) - Test : {} ({})\nNames :\n{}\nConfig :\n{}'.format(
        pred, pred_name, testing, test_name, '\n'.join(names),
        '\n'.join(['- {}\t: {}'.format(k, v) for k, v in default_config.items()])
    ))
    for name in names:
        success = build_and_train(name, allow_retraining, ** default_config)
        
        if testing and success:
            success = test_model(name, test_name, overwrite = overwrite, ** default_config)
        
        if pred and success:
            success = pred_model(name, pred_name, overwrite = overwrite, ** default_config)
        
        if not success:
            break

    
def build_and_train(name, allow_retraining, ** default_config):
    hist = get_model_history(name)

    retraining = False
    if hist is not None and len(hist) > 0:
        logger.info('Model {} has already been trained, {}'.format(
            name, "retraining it for 1 epoch" if allow_retraining else "skipping it."
        ))
        if not allow_retraining: return True
        retraining = allow_retraining
    
        
    if not is_model_name(name):
        err = _run_command('build', ** config_from_name(name, ** default_config))
    
        if err.returncode:
            logger.error('Error when building model {} (status {})'.format(name, err.returncode))
            return False
    
    err = _run_command('train', name, ** training_config_from_name(name, retraining, ** default_config))

    if err.returncode:
        logger.error('Error when training model {} (status code {})'.format(name, err.returncode))
        return False

    logger.info('Successfully built and trained {} !'.format(name))
    return True

def test_model(name, test_name, overwrite = False, ** default_config):
    hist = get_model_history(name)

    suffix = '_EM'
    if 'top5' in test_name: suffix += '-1'
    if hist is None:
        logging.warning('Model {} has not been trained yet, skip its test !'.format(name))
        return True
    elif not is_model_name(name):
        logging.warning('Model {} does not exist, skip its test !'.format(name))
        return True
    elif hist.contains(test_name + suffix):
        if not overwrite:
            logger.info('Test {} for {} already done !'.format(test_name, name))
            return True
        
        logger.info('Overwriting test {}'.format(test_name))
        hist.pop(test_name)
        hist.save()
    
    err = _run_command('test', name, ** testing_config_from_name(name, test_name, ** default_config))

    if err.returncode:
        logger.error('Error when testing model {} (status {})'.format(name, err.returncode))
        return False

    logger.info('Successfully tested {} !'.format(name))
    return True

def pred_model(name, pred_name, overwrite = False, ** default_config):
    hist = get_model_history(name)
    
    map_file    = os.path.join(PRED_DIR, name, pred_name + '.json')
    if hist is None:
        logger.warning('Model {} has not been trained yet, skip its prediction !'.format(name))
        return True
    elif not is_model_name(name):
        logger.warning('Model {} does not exist, skip its prediction !'.format(name))
        return True
    elif os.path.exists(map_file):
        if not overwrite:
            logger.info('Pred {} for {} already done !'.format(pred_name, name))
            return True
        
        logger.info('Overwriting prediction {}'.format(pred_name))
    
    err = _run_command('pred', name, ** predict_config_from_name(
        name, pred_name, overwrite = overwrite, ** default_config
    ))

    if err.returncode:
        logger.error('Error when making prediction for model {} (status {})'.format(
            name, err.returncode
        ))
        return False

    logger.info('Successfully predicted for {} !'.format(name))
    return True
