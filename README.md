# english-vietnamese-machine-translation

## Project Scopes

- **English - Vietnamese Machine Translation**
- **Translation approach:**
    - *Rule-based Machine Translation*
    - *Neural Machine Translation*
- **Evalution metrics:**
    - BLEU score
    - ROUGE score
    - METEOR score

## Dataset

- https://huggingface.co/datasets/vietgpt/open_subtitles_envi
- This dataset contains 3,505,276 rows (177MB)
- 2 columns: {'en', 'vi'}
- EDA
- Data preprocessing

## Models

### 1️⃣ Rule-based Machine Translation

📌 **Reference:** 

- https://arxiv.org/pdf/1708.04559
- https://aclanthology.org/W02-1605.pdf

### 2️⃣Neural Machine Translation

✅ **LSTM (+Attention)**

✅ **Transformer** (can self-modeling and use pretrained models in Hugging Face Hub)

📌 **Reference:** 

- https://arxiv.org/pdf/2012.15515
- https://github.com/facebookresearch/fairseq/tree/main/examples/translation
- https://github.com/auphong2707/machine-translation-en-vi
