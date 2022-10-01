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
import logging
import discord
import numpy as np

from threading import Lock

from loggers import set_level
from models import get_pretrained
from utils import load_json, dump_json, var_from_str
from models.qa import _pretrained, answer_from_web

logger = logging.getLogger(__name__)

set_level('info')

MAX_LENGTH  = 150
ORIGINAL_MODEL  = 'm5_nq_coqa_newsqa_mag_off_entq_ct_wt_ib_2_2_dense'

_emojis_scores  = {
    '1'     : '1️⃣',
    '2'     : '2️⃣',
    '3'     : '3️⃣',
    '4'     : '4️⃣',
    '5'     : '5️⃣',
    '0'     : '❎'
}
_correct_emoji  = {
    'true'  : '✅',
    'false' : '❎'
}
_emojis  = _emojis_scores

_emoji_to_value  = {v : k for k, v in _emojis.items()}

_correctness_key    = 'correctness'
_evaluations    = {
    _correctness_key    : None,
    'quality'   : '**Quality** for Q&A {} \n1 = poor\n3 = correct but not detailed\n5 = well detailed'
}

URL_GIT         = 'https://github.com/yui-mhcp/nlp'
URL_REPORT      = '?'
HELP_MESSAGE    = """
**MAGgie** is a Q&A bot created for the @Mew Master thesis (which is about Q&A models in Deep Learning).

The objective is to test the proposed approach in real-world conditions : try to ask questions and please evaluate the answers by adding reactions !

Commands :
    .help       : get information on commands
    .git        : get the github project's URL
    .report     : get the master thesis' report URL (overleaf)
    .show_model : show model informations
    .ask <question> [OPTIONS]   : ask a question to the bot
        --add_url   : add used urls
        --length_power=<float>      : (value between [0, 1]), encourage longer answers
        --length_temperature=<float>    : (-1 or [0 ... 1]), encourage diversity
        --n         : specifies the number of sites to use (default 5)
        --max_input_texts   : defines the number of paragraphs to use (default to the model's maximum)
    .results    : show statistics on reactions
    .get_url <q_id>     : get used URL for the given question
    .evaluate       : show a random asked question for evaluation

Thank you for your help ! Your evaluation is crucial to show the performance of the technique !
"""

def get_results(directory, min_evaluators = 1):
    """
        Stores results the following way : 
        {
            site_domain : {
                correctness : [...] # top-k, majority vote
                quality     : average_score (float)
            }
        }
    """
    def _is_wiki(urls):
        if not urls: return False
        if isinstance(urls, (list, tuple)):
            return all([_is_wiki(u) for u in urls])
        return urls.split('//')[-1].split('/')[0].endswith('wikipedia.org')
    
    def _take_average(key):
        return True if key != _correctness_key else False
    
    def has_enough_react(infos):
        evaluators = []
        for _, users in infos.items():
            evaluators.extend(users)
        return len(set(evaluators)) >= min_evaluators
    
    results = {}
    questions = {} # domain : question : infos
    
    for q_file in os.listdir(directory):
        data = load_json(os.path.join(directory, q_file))

        if 'question' not in data: continue
        
        url, q = _is_wiki(data['urls']), data['question']
        model = data.get('config', {}).get('model', ORIGINAL_MODEL)
        
        questions.setdefault(model, {})
        questions[model].setdefault(url, {})
        questions[model][url].setdefault(q, {})

        for key, _ in _evaluations.items():
            questions[model][url][q].setdefault(key, {})
            
            for score, val in data.get(key, {}).items():
                questions[model][url][q][key].setdefault(score, []).extend(val)

    
    for model, urls in questions.items():
        results.setdefault(model, {})
        for url, q_infos in urls.items():
            results[model].setdefault(url, {})
            for q, infos in q_infos.items():
                for key, scores in infos.items():
                    if not has_enough_react(scores): continue
                    if _take_average(key):
                        user_scores = {}
                        for score, users in scores.items():
                            if not score.isdigit(): continue
                            for user in users:
                                user_scores.setdefault(user, []).append(int(score))
                        if len(user_scores) > 0:
                            average = np.mean([
                                np.mean(user_score) for _, user_score in user_scores.items()
                            ])

                            results[model][url].setdefault(key, []).append(average)
                    else:
                        majority_score, nb_votes = sorted([
                            (s, len(scores.get(str(s), []))) for s in range(0, 6)
                        ], key = lambda p: p[1], reverse = True)[0]

                        if nb_votes > 0:
                            results[model][url].setdefault(key, {})
                            results[model][url][key].setdefault(majority_score, 0)
                            results[model][url][key][majority_score] += 1
    
    return results

def format_answer(answer, add_url = False, n_passages = 1, max_passage_length = MAX_LENGTH):
    question = answer['question']
    
    logging.info('Formatting answer...')
    des = "Question : **{}**\n".format(question)
    
    model_name = answer.get('config', {}).get('model', None)
    if model_name:
        des += "Model : {}\n".format(model_name)
    if len(answer.get('urls', [])) > 0 and add_url:
        des += "Page(s) used :"
        if isinstance(answer['urls'], (list, tuple)):
            for url in answer['urls']:
                des += '\n - {}'.format(url)
        else:
            des += ' ' + answer['urls']
        des += '\n'
    
    passages = {}
    
    _confidence = ''
    if 'highest_attention_score' in  answer.get('attention_infos', {}):
        _confidence = '(confidence : {:.2f}) '.format(
            answer['attention_infos']['highest_attention_score']
        )
    des += "\nCandidates {}:\n".format(_confidence)
    for i, cand in enumerate(answer['candidates']):
        
        des += "- **#{}{} : {}**\n".format(
            i + 1, ' (log-score : {:.3f})'.format(cand['score']) if 'score' in cand else '', cand['text']
        )
        
        spans = cand.get('attention_infos', {}).get('spans')
        if len(spans)> 0 and n_passages > 0:
            done = 0
            
            des += "  Passages :\n"
            for j, (sent, score, _) in enumerate(spans):
                if sent in passages: continue
                if done >= n_passages: break
                done += 1
                passages[sent] = True
                des += "  -  #{} (attn score {:.3f}) : {}\n".format(
                    j, score, sent if len(sent) < max_passage_length else (sent[:max_passage_length] + ' [...]')
                )
        des += "\n"
    
    des += "\n**Please add reactions for correct answers** and reactions for the **quality** (see next message) ! :smile:"
    
    if len(des) > 2000:
        if n_passages > 0:
            return format_answer(answer, add_url, 0)
        return des[:1990] + ' [...]'
    
    return des


class Maggie(discord.Client):
    def __init__(self, model = None, directory = 'maggie', allowed_config = [], ** kwargs):
        super().__init__(command_prefix = '!', ** kwargs)
        
        self.model  = model if model is not None else _pretrained['en']
        self.directory  = directory
        self.allowed_config = allowed_config + ['add_url']
        
        self.mutex  = Lock()
        self.mutex_react    = Lock()
        
        os.makedirs(self.responses_directory, exist_ok = True)
    
        get_pretrained(self.model)

    @property
    def responses_directory(self):
        return os.path.join(self.directory, 'responses')
    
    @property
    def user_name(self):
        return self.user
    
    @property
    def user_id(self):
        return self.user.id
    
    async def on_ready(self):
        logger.info("{} with ID {} started !".format(self.user_name, self.user_id))
    
    async def on_message(self, context):
        if not (len(context.content) > 0 and context.content[0] in ('.', '!')): return
        
        infos = context.content[1:].split()
        command, msg = infos[0], ' '.join(infos[1:])
        
        logger.info('Get command {}'.format(command))
        
        if not hasattr(self, command):
            await ctx.channel.send('Unknown command : {}\nUse .help for help ;)'.format(command))
            return
        
        await getattr(self, command)(msg, context = context)
    
    async def on_raw_reaction_add(self, reaction):
        user_id = reaction.member.id
        if user_id == self.user_id: return
        
        q_id, eval_type, score = await self.get_infos_from_reaction(reaction)
        if q_id is None: return

        with self.mutex_react:
            filename    = self.get_filename(q_id)
            
            data = load_json(filename)
            
            data.setdefault(eval_type, {})
            if user_id not in data[eval_type].get(score, []):
                logger.info('User {} adds score {} for type {} on question ID {} !'.format(
                    user_id, score, eval_type, q_id
                ))
                data[eval_type].setdefault(score, []).append(user_id)
                
                dump_json(filename, data, indent = 4)
                

    async def on_raw_reaction_remove(self, reaction):
        user_id = reaction.user_id
        if user_id == self.user_id: return
        
        q_id, eval_type, score = await self.get_infos_from_reaction(reaction)
        if q_id is None: return
        
        with self.mutex_react:
            filename    = self.get_filename(q_id)
            
            data = load_json(filename)
            
            data.setdefault(eval_type, {})
            if user_id in data[eval_type].get(score, []):
                logger.info('User {} removes score {} for type {} on question ID {} !'.format(
                    user_id, score, eval_type, q_id
                ))
                data[eval_type][score].remove(user_id)
                
                dump_json(filename, data, indent = 4)
            else:
                logger.error('User {} has no reaction {} for {} on question {}'.format(
                    user_id, score, eval_type, q_id
                ))

    async def help(self, msg, context):
        await context.channel.send(HELP_MESSAGE)
        
    async def hello(self, msg, context):
        await context.channel.send('Hello {} !'.format(context.author.name))
    
    async def git(self, msg, context):
        await context.channel.send(
            'This project is open-source at {} ! :smile:'.format(URL_GIT)
        )
        
    async def report(self, msg, context):
        await context.channel.send(
            'The master thesis\' report is avaialble at {} ! :smile:'.format(URL_REPORT)
        )

    async def show_model(self, msg, context):
        await context.channel.send(str(get_pretrained(self.model)))
    
    async def get_url(self, msg, context):
        if msg + '.json' not in os.listdir(self.responses_directory):
            await context.channel.send('Unknown question : {}'.format(msg))
        else:
            data    = load_json(self.get_filename(msg))
            urls    = data['urls']
            if not isinstance(urls, (list, tuple)): urls = [urls]
            await context.channel.send('URL\'s :\n{}'.format('\n'.join(urls)))
    
    async def ask(self, question, context):
        config = {}
        words = []
        for word  in question.split():
            if word.startswith('--'):
                for c in self.allowed_config:
                    if word.startswith('--{}'.format(c)):
                        config[c] = True if '=' not in word else var_from_str(word.split('=')[1])
                        break
            else:
                words.append(word)
        
        question = ' '.join(words)
        
        add_url = config.pop('add_url', False)
        
        with self.mutex:
            logger.info('Question {} from user {} with config {} !'.format(
                question, context.author, config
            ))
            answer  = answer_from_web(
                question, model = self.model, save = False, analyze_attention = True,
                max_workers = -2, ** config
            )[0]
        
        result = await context.channel.send(format_answer(answer, add_url = add_url))
        q_id = result.id
        logger.info("Question ID : {}".format(q_id))
        
        for eval_type, msg in _evaluations.items():
            answer.setdefault(eval_type, {})
            for score in _emojis_scores.keys():
                answer[eval_type].setdefault(score, [])
        
        dump_json(self.get_filename(q_id), answer, indent = 4)
        
        
        await self.add_default_emojis(result, add_false = True)
        
        for eval_type, msg in _evaluations.items():
            if msg is None: continue
            
            result = await context.channel.send(msg.format(q_id))
            await self.add_default_emojis(result)

    async def evaluate(self, msg, context):
        q_id    = np.random.choice(os.listdir(self.responses_directory)).split('.')[0]
        q_filename  = self.get_filename(q_id)
        
        question    = load_json(q_filename)
        
        des = '**{}** for Q&A {}\n{}'.format(
            _correctness_key.capitalize(), q_id, format_answer(question)
        )[:2000]
        result = await context.channel.send(des)

        await self.add_default_emojis(result, add_false = True)
        
        for eval_type, msg in _evaluations.items():
            if msg is None: continue
            
            result = await context.channel.send(msg.format(q_id))
            await self.add_default_emojis(result)

    
    async def results(self, msg, context):
        res = get_results(self.responses_directory)

        des = ''
        for model, model_infos in res.items():
            des += '\n\n**Model : {}**\n'.format(model)
            for is_restricted, infos in model_infos.items():
                des += '\n# questions (restricted to wiki : {})'.format(is_restricted)
                for key, vals in infos.items():
                    des += '\n- **{}** :'.format(key)
                    if isinstance(vals, dict):
                        total = sum(vals.values())
                        #des += ' - '.join([
                        #    '{} : {} ({:.2f} %)'.format(k, v, 100 * v / total)
                        #    for k, v in vals.items()
                        #])
                        for k, v in sorted(vals.items()):
                            des += '\n  - {} : {} ({:.2f} %)'.format(k, v, 100 * v / total)
                    else:
                        des += ' {:.2f} average score ({} evaluated)'.format(np.mean(vals), len(vals))
        des = des[2:]
        await context.channel.send(des)
    
    async def add_default_emojis(self, ctx, add_false = False):
        for k, emoji in _emojis.items():
            if not add_false and k == '0': continue
            res = await ctx.add_reaction(emoji)
    
    async def get_infos_from_reaction(self, reaction):
        msg_id = reaction.message_id
        if os.path.exists(self.get_filename(msg_id)):
            q_id        = msg_id
            eval_type   = _correctness_key
        else:
            channel = self.get_channel(reaction.channel_id)
            message = await channel.fetch_message(reaction.message_id)
            
            if 'Q&A' not in message.content:
                return None, None, None
                
            q_id    = message.content.split('\n')[0].split()[-1]
            eval_type   = message.content.split()[0].lower().replace('*', '')
            
        return q_id, eval_type, _emoji_to_value.get(reaction.emoji.name, reaction.emoji.name)
    
    def get_filename(self, q_id):
        return os.path.join(self.responses_directory, '{}.json'.format(q_id))


if __name__ == '__main__':
    token = os.environ.get('DISCORD_BOT_TOKEN', None)

    if token is None:
        raise ValueError('You must give the discord bot token as `DISCORD_BOT_TOKEN` env variable !')

    intents = discord.Intents.default()

    bot = Maggie(
        directory = 'maggie', intents = intents,
        allowed_config = ['length_power', 'length_temperature', 'n', 'max_input_texts']
    )
    bot.run(token)