# Brain-Like Language Processing via a Shallow Untrained Multihead Attention Network 

 **Paper Link**: TBD

 <!-- [![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/abs/1234.56789) -->

**Authors**: Badr AlKhamissi, Greta Tuckute, Antoine Bosselut*, Martin Schrimpf*

> \* Equal Supervision

### Abstract
>  Large Language Models (LLMs) have been shown to be effective models of the human language system, with some models predicting most explainable variance of brain activity in current datasets. Even in untrained models, the representations induced by architectural priors can exhibit reasonable alignment to brain data. In this work, we investigate the key architectural components driving the surprising alignment of untrained models. To estimate LLM-to-brain similarity, we first select language-selective units within an LLM, similar to how neuroscientists identify the language network in the human brain. We then benchmark the brain alignment of these LLM units across five different brain recording datasets. By isolating critical components of the Transformer architecture, we identify tokenization strategy and multihead attention as the two major components driving brain alignment. A simple form of recurrence further improves alignment. We further demonstrate this quantitative brain alignment of our model by reproducing landmark studies in the language neuroscience field, showing that localized model units -- just like language voxels measured empirically in the human brain -- discriminate more reliably between lexical than syntactic differences, and exhibit similar response profiles under the same experimental conditions. Finally, we demonstrate the utility of our model's representations for language modeling, achieving improved sample and parameter efficiency over comparable architectures. Our model's estimates of surprisal sets a new state-of-the-art in the behavioral alignment to human reading times. Taken together, we propose a highly brain- and behaviorally-aligned model that conceptualizes the human language system as an untrained shallow feature encoder, with structural priors, combined with a trained decoder to achieve efficient and performant language processing.

## Repository Structure

### Functional Localization 

- **Step 1**: Download language localization stimuli set from [this link](https://www.dropbox.com/sh/c9jhmsy4l9ly2xx/AACQ41zipSZFj9mFbDfJJ9c4a?e=1&dl=0) into the following folder `language-localization/fedorenko10_stimuli`.
- **Step 2**: See `language-localization/scripts/run_localization.sh` for example on how to run functional localization for pretrained and untrained LLaMA-2-7B. 

### Brain Score

- **Step 1**: Clone `https://github.com/bkhmsi/brain-score-language` in root directory of repository.
- **Step 2**: Follow setup instructions in the `brain-score-language` repository `README.md`
- **Step 2**: See `brain-alignment/scripts` for examples of measuring brain alignment of different model configurations on the 5 benchmarks.

## Citation
TBD