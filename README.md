# :yum: Natural Language Processing (NLP) & Natural Language Understanding (NLU)

**Important note** : this github is currently a simple copy of the original Master thesis' repository from @Ananas120 ([link](https://github.com/Ananas120/mag)). This repo currently only contains code for `Question Answering (Q&A)`. In the next update, it will be improved to be more general and contains more NLP tasks such as `Masked Language Modeling (MLM)`, `Next Word Prediction (NWP)`, etc. 

## Project structure

```bash
├── custom_architectures/   : custom architectures
│   ├── transformers_arch/  : specific blocks for Transformers (BERT / BART / GPT-2 / ...)
├── custom_layers/          : custom layers
├── custom_train_objects/   : custom objects for training
├── datasets/               : utilities for dataset loading / processing
├── hparams/                : utility class to define modulable hyper-parameters
├── loggers/                : some logging utilities
├── models/                 : main `BaseModel` subclasses directory
│   ├── interfaces/         : directory for `BaseModel` class and useful interfaces\*
│   ├── qa/                 : directory for `Q&A` classes
├── pretrained_models/      : saving directory for pretrained models
├── unitest/                : custom unitest framework to test models' consistency
└── utils/                  : utilities for data processing

```

See [my data_processing repository](https://github.com/yui-mhcp/data_processing) for more information on the `utils` module and `data processing` features, as well as `loggers` and `unitest`.

See [my base project](https://github.com/yui-mhcp/base_dl_project) for more information on the `BaseModel` class, supported datasets, project extension, ...


## Available features

- **Text-To-Speech** (module `models.tts`) :

| Feature   | Fuction / class   | Description |
| :-------- | :---------------- | :---------- |
| Text-To-Speech    | `tts`             | perform TTS on text you want with the model you want  |
| stream            | `tts_stream`      | perform TTS on text you enter |
| TTS logger        | `loggers.TTSLogger`   | converts `logging` logs to voice and play it |

You can check the `text_to_speech` notebook for a concrete demonstration

## Available models

### Model architectures

Available architectures : 
- `Synthesizer` :
    - [Tacotron2](https://arxiv.org/abs/1712.05884) with extensions for multi-speaker (by ID or `SV2TTS`)
    - [SV2TTS](https://papers.nips.cc/paper/2018/file/6832a7b24bc06775d02b7406880b93fc-Paper.pdf) extension of the Tacotron2 architecture for multi-speaker based on speaker's embeddings\*
- `Vocoder` :
    - [Waveglow](https://arxiv.org/abs/1811.00002)

\* The speaker's embeddings are created with the Siamese Networks approach, which differs from the original paper. Check the [Siamese Networks](https://github.com/yui-mhcp/siamese_networks) project for more information on this architecture.

### Model weights

| Task      | Language  | Name      | Class         | Dataset   | Trainer   | Weights   |
| :-------: | :-------: | :-------: | :-----------: | :-------: | :-------: | :-------: |
| /         | /         | /         | /             | /         | /         | /         |

Weights will be added in the next update

Models must be unzipped in the `pretrained_models/` directory !

## Installation and usage

1. Clone this repository : `git clone https://github.com/yui-mhcp/nlp.git`
2. Go to the root of this repository : `cd nlp`
3. Install requirements : `pip install -r requirements.txt`
4. Open `text_to_speech` notebook and follow the instruction !

## TO-DO list :

- [x] Make the TO-DO list
- [ ] Clean-up the code
- [ ] Comment the code
- [ ] Extend for general NLU tasks (and not only Q&A)
    - [ ] Add Masked Language Modeling (MLM) support
    - [ ] Add Next Word Prediction (NWP) support
    - [ ] Add Neural Machine Translation (NMT) support
    - [ ] Add text classification tasks (such as intent / emotion / topic classification)
- [ ] Add pretrained models (for Q&A in English) (Ananas120's master thesis models)
- [ ] Add new languages support
- [ ] Add document parsing to perform Q&A on document (in progress)

## NLP vs NLU

`Natural Language Processing (NLP)` and `Natural Language Understanding (NLU)` are general terms that groups a bunch of tasks related to language. Both are mainly used the same way but in theory, NLP is larger than NLU as it groups both NLU and speech-related tasks, such as `Text-To-Speech (TTS)` and `Speech-To-Text (STT)`, while NLU more reflects text *understanding* tasks (MLM, NMT, NWP, Q&A, text-classification, ...). 
For this reason, it is possible that this repo will be duplicated into a `nul/` repository and this one will integrate `nlu` as well as existing [TTS](https://github.com/yui-mhcp/text_to_speech) and [STT](https://github.com/yui-mhcp/speech_to_text) repositories. 

Furthermore, the term *understanding* is an exageration as models do not really *understands* the language / the concepts behind words but mimic what they have learned to do / what contains their training database. 

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

All the citations for the master thesis are available in the [CITATIONS](CITATIONS.thesis.bib) file with links for papers. 

The thesis report is not published by the university yet. 