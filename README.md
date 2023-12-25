# :yum: Natural Language Processing (NLP) & Natural Language Understanding (NLU)

This github is an extension of the [@Ananas120 Master thesis' repository](https://github.com/Ananas120/mag), extending [my base project](https://github.com/yui-mhcp/base_dl_project) to Q&A. I have generalized and cleaned up his code to allow general NLP tasks (and not only Q&A). Thanks to him for his contribution ! :smile:

Check the [CHANGELOG](https://github.com/yui-mhcp/yui-mhcp/blob/main/CHANGELOG.md) file to have a global overview of the latest modifications ! :yum:

**IMPORTANT NOTE** : This project is currently updated, meaning that multiple features may not currently work ! They will be updated in the near future with new fancy applications and models ! :smile:

## Project structure

```bash
├── custom_architectures
├── custom_layers
├── custom_train_objects
│   ├── losses
│   │   └── qa_retriever_loss.py    : special loss for AnswerRetriever model
│   ├── metrics
│   │   ├── f1.py           : F1 implementation as a `tf.keras.metrics.Metric` class
│   │   └── top_k_f1.py     : extension to support Beam-Search output
├── datasets
├── hparams
├── loggers
├── models
│   ├── nlu             : general NLU classes
│   │   ├── base_nlu_generator.py   : extension of `BaseNLUModel` for text-generative models
│   │   ├── base_nlu_model.py       : general interface defining data loading for text-based models
│   │   └── nlu_utils.py            : utilities for the NLU models
│   ├── qa              : directory for Q&A based models
│   │   ├── answer_generator.py     : model that generates an answer
│   │   ├── answer_retriever.py     : model that retrieves the answer within the context
│   │   ├── mag.py                  : extension of `AnswerGenerator` to support MAG-style
│   │   ├── question_generator.py   : model that generates a question based on an answer
│   │   └── web_utils.py            : utilities to search on the web and parse results (for Q&A inputs)
├── pretrained_models
├── unitest
├── utils
├── CITATIONS.thesis.bib    : citations for the master thesis
├── Dockerfile-maggie       : runs the maggie bot in a Docker container
├── Makefile                : defines commands to run / stop maggie
├── docker-compose-maggie.yml   : runs maggie in docker-compose
├── example_answer_generator.ipynb
├── example_mag.ipynb
├── experiments.py          : abstract file defining functions to run multiple experiments
├── experiments_mag.py      : defines the functions to run MAG experiments
├── maggie.py               : the code for the MAGgie bot
├── main.py                 : main file to run build / train / test / predict command-line
└── question_answering.ipynb
```

Check [the main project](https://github.com/yui-mhcp/base_dl_project) for more information about the unextended modules / structure / main classes. 


## Available features

- **Question Answering** (module `models.qa`) :

| Feature   | Fuction / class   | Description |
| :-------- | :---------------- | :---------- |
| Q&A       | `answer_from_web` | Performs question-answering based on top-k most relevant web pages |

You can check the `question_answering` notebook for a concrete demonstration

## Available models

### Model architectures

Available architectures :
- [BERT](http://arxiv.org/abs/1810.04805)
- [BART](https://aclanthology.org/2020.acl-main.703)
- [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)
- `MAG`     : a general wrapper for text-based Transformers. 


### Model weights

| Task      | Name      | Lang  | Class | Dataset   | Trainer   | Weights   |
| :-------: | :-------: | :---: | :---: | :-------: | :-------: | :-------: |
| Q&A       | maggie    | en    | MAG   | `NQ`, `CoQA`, `NewsQA`| [Ananas120](https://github.com/Ananas120) | [Google Drive](https://drive.google.com/file/d/1koG-UMMz8557zjkifTCpQMBgWVCqr1XS/view?usp=sharing)  |

Weights will be added in the next update

Models must be unzipped in the `pretrained_models/` directory !

## Installation and usage

1. Clone this repository : `git clone https://github.com/yui-mhcp/nlp.git`
2. Go to the root of this repository : `cd nlp`
3. Install requirements : `pip install -r requirements.txt`
4. Open `question_answering` notebook and follow the instruction !

## TO-DO list :

- [x] Make the TO-DO list
- [x] Clean-up the code
- [x] Comment the code
- [x] Create general NLU classes
- [x] Allow to modify the default configuration in `predict` method (experimental)
- [ ] Extend for general NLU tasks (and not only Q&A)
    - [ ] Add Masked Language Modeling (MLM) support
    - [ ] Add Next Word Prediction (NWP) support
    - [ ] Add Neural Machine Translation (NMT) support
    - [ ] Add text classification tasks (such as intent / emotion / topic classification)
- [x] Add pretrained models (for Q&A in English) (Ananas120's master thesis models)
- [ ] Add new languages support
- [ ] Add document parsing to perform Q&A on document (in progress)
- [ ] Convert the `llama2` model to `tensorflow`

## NLP vs NLU

`Natural Language Processing (NLP)` and `Natural Language Understanding (NLU)` are general terms that groups a bunch of tasks related to language. Both are mainly used the same way but in theory, NLP is larger than NLU as it groups both NLU and speech-related tasks, such as `Text-To-Speech (TTS)` and `Speech-To-Text (STT)`, while NLU more reflects text *understanding* tasks (MLM, NMT, NWP, Q&A, text-classification, ...). 
For this reason, it is possible that this repo will be duplicated into a `nul/` repository and this one will integrate `nlu` as well as existing [TTS](https://github.com/yui-mhcp/text_to_speech) and [STT](https://github.com/yui-mhcp/speech_to_text) repositories. 

Furthermore, the term *understanding* is an exageration as models do not really *understands* the language / the concepts behind words but mimic what they have learned to do / what contains their training database. 

## Pipeline-based prediction

The `BaseNLUModel` (and its subclasses) model supports the pipeline-based prediction, meaning that all the tasks you see in the below graph are multi-threaded. Check the [data_processing project](https://github.com/yui-mhcp/data_processing) for a better understanding of the `producer-consumer` framework. 

![NLP pipelinepipeline](nlp_pipeline.jpg)

## Contacts and licence

You can contact [me](https://github.com/yui-mhcp) at yui-mhcp@tutanota.com or on [discord](https://discord.com) at `yui#0732`

The objective of these projects is to facilitate the development and deployment of useful application using Deep Learning for solving real-world problems and helping people. 
For this purpose, all the code is under the [Affero GPL (AGPL) v3 licence](LICENCE)

All my projects are "free software", meaning that you can use, modify, deploy and distribute them on a free basis, in compliance with the Licence. They are not in the public domain and are copyrighted, there exist some conditions on the distribution but their objective is to make sure that everyone is able to use and share any modified version of these projects. 

Furthermore, if you want to use any project in a closed-source project, or in a commercial project, you will need to obtain another Licence. Please contact me for more information. 

For my protection, it is important to note that all projects are available on an "As Is" basis, without any warranties or conditions of any kind, either explicit or implied. However, do not hesitate to report issues on the repository's project or make a Pull Request to solve it :smile: 

If you use this project in your work, please add this citation to give it more visibility ! :yum:

```
@misc{yui-mhcp
    author  = {yui},
    title   = {A Deep Learning projects centralization},
    year    = {2021},
    publisher   = {GitHub},
    howpublished    = {\url{https://github.com/yui-mhcp}}
}
```

## Acknowledgments

Thanks to @Ananas120 for his contribution and sharing his code ! 

## Notes and references

All the citations for the master thesis are available in the [CITATIONS](CITATIONS.thesis.bib) file with links for papers. They can be good papers to start NLP and discovers the Transformers-based models which are omnipresent nowadays in NLP /  NLU !

The thesis report is not published by the university yet. 
