# **English-to-Vietnamese Machine Translation**

This repository contains the implementation of a machine translation system design to translate text from English to Vietnamese, developed as part of a Natural Language Processing project. The project explores three machine translation paradigms: Rule-Based Machine Translation (RBMT), Statistical Machine Translation (SMT), and Neural Machine Translation (NMT).

## **Project Overview**

This project aims to develop a system for translating English to Vietnamese, tackling challenges like differing word orders, Vietnamese's tonal nature, and lack of inflection. It compares RBMT, SMT, and NMT models, evaluating their performance using metrics such as BLEU, ROUGE, METEOR, COMET, and BERTScore.

## **Dataset**

The dataset usedd is `ncduy/mt-en-vi` from the Hugging Face Hub, consisting of parallel English-Vietnamese sentences. Key characteristics: 
* **Columns Used:** English `en` and Vietnamese `vi` sentence pairs.
* **Sentence Lengths:** Most sentences are short (0-50 words), with medians of 10-20 words (English) and 20-30 words (Vietnamese). Outliers reach up to 1,200 (English) and 1,400 (Vietnamese) words.
* **Linguistic Patterns:** English emphasizes semantic content (e.g., "one", "time"), while Vietnamese focuses on syntactic structure (e.g., "là", "và"). Bigrams and trigrams highlight prepositional phrases in English and modal verbs in Vietnamese.

Download the re-splitted dataset: [MT Dataset](https://husteduvn-my.sharepoint.com/:f:/g/personal/thanh_lh225458_sis_hust_edu_vn/EjAIwPiRrv5FvVhBlRwUt28B4fkWE8m1uTTA9ggHncsW8Q?e=g751uM) and then place this in the `dataset/` directory.

## **Methods and Models**
### **Pre-processing**
* **General:** Remove extra spaces, HTML tags, URLs
* **English:** Simplify non-ASCII characters (e.g., 'é' -> 'e')
* **Vietnamese:** Validate Vietnamese format

### **Models**
**Rule-Based Machine Translation (RBMT):** 
  * Implements Transfer-Based Machine Translation (TBMT) with handcrafted grammatical, syntactic, and semantic rules.
  * Stages: Source analysis (POS tagging, syntactic parsing), transfer grammar (reordering English to Vietnamese syntax), and generation (lexical transfer, tone/aspect/mood mapping)
  * Strengths: High accuracy for well-defined grammars; transparent and adaptable.
  * Weaknesses: Time-intensive rule crafting; struggles with idiomatic expressions.

**Statistical Machine Translation (SMT):** 
  * Uses probabilistic modeling with translation and language models based on Bayes' theorem.
  * Key components: Word alignment (IBM Model 1, EM algorithm), phrase-based translation, and n-gram language model with Laplace smoothing.
  * Pipeline: Preprocessing, word segmentation (ViTokenizer), alignment, phrase table creation, and decoding via beam search.
  * Challenges: Word order differences, limited parallel corpora, and handling idiomatic expressions.

**Neural Machine Translation (NMT):** 
  * mT5: A text-to-text Transformer pretrained on mC4, fine-tuned with a task prefix (e.g., "translate English to Vietnamese:"). Versatile but less specialized for translation.
  * mBART50: A Transformer-based model pretrained with denoising autoencoding, using language-specific tokens (e.g., `<en_XX>`, `<vi_VN>`). Tailored for translation, excelling in zero-shot and few-shot scenarios.
  * Both models handle Vietnamese tone and flexible syntax effectively due to multilingual pretraining.

## **Results**

| Model       | BLEU   | ROUGE-1 | ROUGE-2 | ROUGE-L | METEOR | COMET | BERTScore |
|-------------|--------|---------|---------|---------|--------|-------|-----------|
| RBMT        | 0.0783 | 0.5723  | 0.2806  | 0.4518  | 0.3773 | 0.5577 | 0.7726    |
| SMT         | 0.0890 | 0.5134  | 0.2635  | 0.4278  | 0.3475 | 0.5393 | 0.7748    |
| Finetuned-mT5        | 0.1841 | 0.5662  | 0.2985  | 0.4747  | 0.3972 | 0.6844 | 0.8241    |
| Finetuned-mBART50    | 0.3337 | 0.6729  | 0.4540  | 0.6032  | 0.5610 | 0.8237 | 0.8615    |

- **Finetuned-mBART50** outperformed all models, excelling in fluency, structural coherence, and semantic accuracy.
- **Finetuned-mT5** showed significant improvements over traditional models but was less effective than FmBART50.
- **RBMT** and **SMT** struggled with low BLEU scores and limited fluency, highlighting their limitations for complex translations.

## **Conclusion**
The project successfully developed and compared RBMT, SMT, and NMT models for English-to-Vietnamese translation. FmBART50 demonstrated superior performance, leveraging multilingual pretraining and language-specific tokens to handle Vietnamese's linguistic nuances. Future work could focus on:
- Expanding the dataset to include diverse domains and longer sentences.
- Exploring hybrid RBMT-SMT, RBMT-NMT approaches.
- Developing lightweight NMT models for low-resource settings.

## **Installation**

Clone the repository and navigate to the project directory:
```bash
git clone https://github.com/fisherman611/english-vietnamese-machine-translation.git
```

Navigate to the project directory: 
```bash 
cd english-vietnamese-machine-translation
```

Install the required dependencies:
```bash
pip install -r requirements.txt
```
## **Download the pretrained model**
Download the pretrained model checkpoints from this [OneDrive link]((https://husteduvn-my.sharepoint.com/:f:/g/personal/thanh_lh225458_sis_hust_edu_vn/EjAIwPiRrv5FvVhBlRwUt28B4fkWE8m1uTTA9ggHncsW8Q?e=g751uM))

Place the downloaded checkpoint in the `checkpoint/` directory within the repository.

## **Inference**

Run the following command, you can select the name of model (`rbmt`, `smt`, `mbart50`, `mt5`) and type your text you want:
```bash 
python infer.py --model_type <type name of model here> --text "<type text you want here>"
```

## **References:**
[1]  A. Ahsan, P. Kolachina, S. Kolachina, D. Sharma, and R. Sangal. Coupling statistical machine translation with rule-based transfer and generation. AMTA 2010- 9th Conference of the Association for Machine Translation in the Americas, 01 2010. URL https://aclanthology.org/2010.amta-papers.6.pdf.

[2]  P. F. Brown, S. A. D. Pietra, V. J. D. Pietra, and R. L. Mercer. The mathematics of statistical machine translation: Parameter estimation. Computational Linguistics, 19(2):263–311, 1993. URL https://aclanthology.org/J93-2003/.

[3]  M. Honnibal and I. Montani. spacy 2: Natural language understanding with bloom embeddings, convolutional neural networks and incremental parsing. https://spacy.io, 2017. To appear.

[4] P. Koehn. Chapter 5: Phrase-based models. https://www2.statmt.org/book/slides/05-phrase-based-models.pdf, 2009. Lecture slides from the book "Statistical Machine Translation".

[5] L. H. Phuong. Vitokenizer: Vietnamese word segmentation tool. https://github.com/letuananh/vn-tokenizer, 2007. Available from VLSP resources.

[6] S. S. Statistical vs rule based machine translation; a case study on indian language perspective. https://arxiv.org/abs/1708.04559, 2017. arXiv:1708.04559 [cs.CL].

[7] Y. Tang, C. Tran, X. Li, P.-J. Chen, N. Goyal, V. Chaudhary, J. Gu, and A. Fan. Multilingual translation with extensible multilingual pretraining and finetuning. Transactions of the Association for Computational Linguistics (TACL), 9:102–118, 2021. URL https://arxiv.org/abs/2008.00401.

[8]  L. C. Thompson. Vietnamese reference grammar. University of Hawaii Press, 1985. Discusses the topic-comment structure and syntactic characteristics of Vietnamese.

[9]  T. V. Trung. Pyvi: Vietnamese word tokenizer. https://github.com/trungtv/pyvi, 2018. GitHub repository.

[10]  A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin. Attention is all you need, 2023. URL https://arxiv.org/abs/1706.03762.

[11]  L. Xue, N. Constant, A. Roberts, M. Kale, R. Al-Rfou, A. Siddhant, A. Barua, and C. Raffel. mt5: A massively multilingual pre-trained text-to-text transformer. Proceedings of NAACL 2021, 2021. URL https://arxiv.org/abs/2010.11934


## **Contributors**
- [Lương Hữu Thành](https://github.com/fisherman611) - 20225458
- [Vũ Trung Thành](https://github.com/thanh309) - 20220066
- [Nguyễn Mậu Trung](https://github.com/Pearlcentt) - 20225534
- [Đoàn Anh Vũ](https://github.com/bluff-king) - 20225465

## **License** 
This project is licensed under the [MIT License](LICENSE).