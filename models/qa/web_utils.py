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

import json
import urllib
import logging
import requests
import async_timeout

from utils.text import _wiki_cleaner

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
    
    logging.debug('URL\'s : {}'.format(url))
    
    kept_urls   = []
    parsed      = []
    
    for url_i in url:
        if len(kept_urls) >= n: break
        try:
            res = requests.get(url_i, timeout = timeout)
            if res.status_code != 200 or 'html' not in res.headers.get('Content-Type', ''):
                logging.dev('Skip url (content type = {}, status = {})'.format(
                    res.headers.get('Content-Type', ''), res.status_code
                ))
                continue
            
            page = parse_html(res.text, ** parser_config)
            if len(page) <= min_page_length:
                logging.dev('Page {} is too short ({})'.format(url_i, len(page)))
                continue
            
            kept_urls.append(url_i)
            parsed.extend(page)
        except Exception as e:
            logging.error('Error when getting url {} : {}'.format(url_i, e))
    
    if len(parsed) == 0: parsed = [{'title' : '', 'text' : '<no result>'}]
    
    logging.dev('Kept url(s) : {}'.format(kept_urls))
    
    data = {
        'question'      : question,
        'paragraphs'    : [p['text'] for p in parsed],
        'titles'        : [p['title'] for p in parsed],
        'engine'        : engine,
        'urls'          : kept_urls[0] if len(kept_urls) == 1 else kept_urls,
        'config'        : kwargs
    }
    return data

def search_on_web(query, * args, engine = None, test_all_engines = True,
                  timeout = DEFAULT_TIMEOUT, ** kwargs):
    global _default_engine
    
    if engine is None: engine = _default_engine

    if engine not in _search_engines:
        raise ValueError('Unknown search engine !\n  Accepted : {}\n  Got : {}'.format(
            tuple(_search_engines.keys()), engine
        ))
    
    if test_all_engines:
        engines  = [engine] + [e for e in _search_engines.keys() if e != engine]
    else:
        engines = [engine]
    for engine_i in engines:
        try:
            logging.info('Try query {} on engine {}...'.format(query, engine_i))
            try:
                with async_timeout.timeout(timeout):
                    urls    = _search_engines[engine_i](query, * args, ** kwargs)
            except RuntimeError as e:
                logging.warning('Cannot use timeout here : {} !'.format(e))
                urls    = _search_engines[engine_i](query, * args, ** kwargs)
            if len(urls) == 0:
                logging.warning('No result with engine {} for query {}, trying another search engine !'.format(engine_i, query))
                continue
            
            result = {'engine' : engine_i, 'urls' : urls}
            
            if _default_engine is None: _default_engine = engine_i
            
            return result
        except Exception as e:
            print(type(e))
            logging.error('Error with engine {} : {}, trying next engine'.format(engine_i, str(e)))
            #if _default_engine == engine: _default_engine = None
    
    return {'engine' : None, 'urls' : []}

def search_on_google(query, n = 10, site = None, ** kwargs):
    """ Return a list of url's for a given query """
    import googlesearch as google
    
    if site is not None: query = '{} site:{}'.format(query, site)
    
    results = []
    for res in google.search(query, safe = 'on', ** kwargs):
        results.append(res)
        if len(results) == n: break
    
    return results

def search_on_ddg(query, n = 10, site = None, ** kwargs):
    if site is not None: query = '{} site:{}'.format(query, site)

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

def search_on_bing(query, n = 10, site = None, ** kwargs):
    from bs4 import BeautifulSoup
    
    if site is not None: query = query + ' site:' + site
    params = {
        'q'     : '+'.join(query.split())
    }
    encoded = '&'.join(['{}={}'.format(k, v) for k, v in params.items()])
    url = '{}?{}'.format(_bing_api_url, encoded)
    res = BeautifulSoup(requests.get(url, headers = {'User-Agent' : 'mag'}).text)

    raw_results = res.find_all('li', attrs= {'class' : 'b_algo'})
    links = []
    for raw in raw_results:
        link = raw.find('a').get('href')
        if link: links.append(link)
        
    return links[:n]

_search_engines = {
    'google'    : search_on_google,
    'bing'      : search_on_bing,
    'ddg'       : search_on_ddg
}
