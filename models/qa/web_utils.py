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

import json
import urllib
import logging
import requests
import async_timeout

from utils.text import _wiki_cleaner

logger = logging.getLogger(__name__)

_default_parser_config  = {
    'tags_to_keep'  : ['p'],
    'skip_header'   : True,
    'skip_footer'   : True,
    'skip_aside'    : True,
    'skip_hrefs'    : True,
    'remove_pattern'    : _wiki_cleaner
}

_default_engine = 'google'
DEFAULT_TIMEOUT = 10

_ddg_api_url    = 'http://api.duckduckgo.com/'
_bing_api_url   = 'http://www.bing.com/search'

def _search_wrapper(fn):
    def wrapper(query, * args, site = None, n = 10, ** kwargs):
        if site: query = '{} site:{}'.format(query)
        
        urls = []
        for url in fn(query, * args, ** kwargs):
            urls.append(url)
            if len(urls) >= n: break
        return urls
    
    wrapper_fn = wrapper
    wrapper_fn.__doc__  = fn.__doc__
    wrapper_fn.__name__ = fn.__name__
    
    return wrapper_fn

def prepare_web_data(question,
                     url     = None,
                     parser_config       = _default_parser_config,

                     engine  = None,
                     test_all_engines    = True,
                     n       = 5,
                     timeout = DEFAULT_TIMEOUT,
                     site    = None,
                     min_page_length = 1,
                     ** kwargs
                    ):
    from utils.text import parse_html
    
    if url is None:
        result = search_on_web(
            question, n = n * 2, site = site, engine = engine, test_all_engines = test_all_engines
        )
        url, engine = result['urls'], result['engine']
    if not isinstance(url, (list, tuple)): url = [url]
    
    logger.debug('URL\'s : {}'.format(url))
    
    kept_urls   = []
    parsed      = []
    
    for url_i in url:
        if len(kept_urls) >= n: break
        try:
            res = requests.get(url_i, timeout = timeout)
            if res.status_code != 200 or 'html' not in res.headers.get('Content-Type', ''):
                logger.dev('Skip url (content type = {}, status = {})'.format(
                    res.headers.get('Content-Type', ''), res.status_code
                ))
                continue
            
            page = parse_html(res.text, ** parser_config)
            if len(page) <= min_page_length:
                logger.dev('Page {} is too short ({})'.format(url_i, len(page)))
                continue
            
            kept_urls.append(url_i)
            parsed.extend(page)
        except Exception as e:
            logger.error('Error when getting url {} : {}'.format(url_i, e))
    
    if len(parsed) == 0: parsed = [{'title' : '', 'text' : '<no result>'}]
    
    logger.dev('Kept url(s) : {}'.format(kept_urls))
    
    data = {
        'question'      : question,
        'paragraphs'    : [p['text'] for p in parsed],
        'titles'        : [p['title'] for p in parsed],
        'engine'        : engine,
        'urls'          : kept_urls[0] if len(kept_urls) == 1 else kept_urls,
        'config'        : kwargs
    }
    return data

def search_on_web(query,
                  * args,
                  engine    = None,
                  timeout   = DEFAULT_TIMEOUT,
                  test_all_engines  = True,
                  ** kwargs
                 ):
    """
        Searches `query` with (possibly) multiple search engine API and returns results
        
        Arguments :
            - query : the search query
            - engine    : the search engine's name to use
            - timeout   : the API request timeout
            - test_all_engines  : whether to test other engines if the given onee fails
            - args / kwargs : forwarded to the search function, you can check common kwargs with `help(_search_wrapper)` which wraps all search functions to preprocess the query and filters the results
        Returns : a dict {engine : engine_name, urls : list_of_urls}
            - engine_name   : the engine that returned the urls
            - list_of_urls  : urls returned by the search engine's API
    """
    global _default_engine
    
    if engine is None: engine = _default_engine

    if engine not in _search_engines:
        raise ValueError('Unknown search engine !\n  Accepted : {}\n  Got : {}'.format(
            tuple(_search_engines.keys()), engine
        ))
    
    other_engines = [] if not test_all_engines else [e for e in _search_engines if e != engine]
    engines = [engine] + other_engines

    for engine_i in engines:
        try:
            logger.info('Try query {} on engine {}...'.format(query, engine_i))
            try:
                with async_timeout.timeout(timeout):
                    urls    = _search_engines[engine_i](query, * args, ** kwargs)
            except RuntimeError as e:
                logger.warning('Cannot use async_timeout here : {} !'.format(e))
                urls    = _search_engines[engine_i](query, * args, ** kwargs)
            
            if len(urls) == 0:
                logger.warning('No result with engine {} for query {}, trying another search engine !'.format(engine_i, query))
                continue
            
            if _default_engine is None: _default_engine = engine_i

            return {'engine' : engine_i, 'urls' : urls}
        except Exception as e:
            logger.error('Error with engine {} : {}, trying next engine'.format(engine_i, str(e)))
    
    logger.warning('No engine succeed !')
    return {'engine' : None, 'urls' : []}

@_search_wrapper
def _search_on_google(query, ** kwargs):
    """ Return a list of url's for a given query """
    import googlesearch as google
    
    return google.search(query, safe = 'on', ** kwargs)

@_search_wrapper
def _search_on_ddg(query, ** kwargs):
    params = {
        'q'     : query,
        'o'     : 'json',
        'kp'    : '1',
        'no_redirect'   : '1',
        'no_html'       : '1'
    }

    url = '{}?{}'.format(_ddg_api_url, urllib.parse.urlencode(params))
    res = requests.get(url, headers = {'User-Agent' : 'mag'})
    
    if len(res.content) == 0 or not res.json()['AbstractURL']: return []
    return res.json()['AbstractURL']

@_search_wrapper
def _search_on_bing(query, user_agent = None, ** kwargs):
    from bs4 import BeautifulSoup
    
    params = {
        'q'     : '+'.join(query.split())
    }
    encoded = '&'.join(['{}={}'.format(k, v) for k, v in params.items()])
    url = '{}?{}'.format(_bing_api_url, encoded)
    res = BeautifulSoup(requests.get(url, headers = {'User-Agent' : user_agent}).text)

    for raw in res.find_all('li', attrs = {'class' : 'b_algo'}):
        link = raw.find('a').get('href')
        if link: yield link

_search_engines = {
    'google'    : _search_on_google,
    'bing'      : _search_on_bing,
    'ddg'       : _search_on_ddg
}
