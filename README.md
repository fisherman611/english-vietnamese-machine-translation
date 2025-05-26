# **English-Vietnamese Machine Translation**

## 📌 Project Overview

This project aims to build an effective **English-to-Vietnamese Machine Translation System**, evaluating the performance of three major paradigms in MT:

- **Rule-Based Machine Translation (RBMT)**
- **Statistical Machine Translation (SMT)**
- **Neural Machine Translation (NMT)**

Our objective is to implement, train, and assess these models using reliable evaluation metrics.

---

## 🔬 Scope and Methodologies

### 🧠 Translation Approaches

#### 1. Rule-Based Machine Translation (RBMT)
- Implements **Transfer-Based Machine Translation (TBMT)** using handcrafted syntactic, grammatical, and semantic rules.
- Workflow: POS Tagging → Syntactic Parsing → Transfer Grammar → Lexical Transfer → Generation
- Strength: Transparent, rule-driven system.
- Limitation: Costly and less adaptable.

#### 2. Statistical Machine Translation (SMT)
- Phrase-based SMT model trained on aligned bilingual corpora.
- Translation generated using: `argmax_E P(E|V) P(V)`.

#### 3. Neural Machine Translation (NMT)
- Encoder-Decoder architecture with Attention.
- Models used: LSTM, RNN, Transformers with BERT, GPT, mBART50, T5 (via Hugging Face).
- Improvements: Pointer-Generator, Coverage Vector, Beam Search.

---

## 🧪 Evaluation Metrics

| Metric       | Description                                                       |
|--------------|-------------------------------------------------------------------|
| **BLEU**     | N-gram precision; penalizes overly short outputs.                 |
| **ROUGE**    | Recall-based metric; useful for longer sequences.                 |
| **METEOR**   | Uses synonyms/stemming; better correlates with human evaluation. |
| **COMET**    | Embedding-based; evaluates source, reference & hypothesis.        |
| **BERTScore**| Embedding similarity using BERT token vectors.                    |

---

## 📥 Input & 📤 Output

- **Input**: English sentence  
  _Example_: `"Hello, how are you?"`

- **Output**: Vietnamese translation  
  _Output_: `"Xin chào, bạn có khỏe không?"`

---

## 📚 Dataset

**Hugging Face Dataset:** [ncduy/mt-en-vi](https://huggingface.co/datasets/ncduy/mt-en-vi)

| Split       | # Examples  |
|-------------|-------------|
| Train       | 2,884,451   |
| Validation  | 11,316      |
| Test        | 11,225      |

Each record includes:
- `en`: English sentence
- `vi`: Vietnamese sentence
- `source`: Original dataset source

---

## 🔄 Preprocessing Pipeline

- Normalize Unicode and diacritics
- Remove HTML, URLs, redundant spaces
- Named Entity Recognition using spaCy
- Tokenization and case normalization
- Data cleaning tailored for each model type

---

## 🤖 Models

### ✅ Rule-Based (RBMT)
- Rule templates based on Vietnamese grammar structure
- CFG parsing, Transfer grammar, TAM mapping

### ✅ Statistical (SMT)
- IBM alignment models
- Phrase table & language model
- Decoded using a noisy channel formulation

### ✅ Neural (NMT)
- Encoder-Decoder with attention
- Pretrained: BERT, GPT, mBART50, T5
- Beam Search, Coverage, and Pointer Generator

---

## 📊 Results

- NMT provides superior fluency and context preservation
- SMT performs well on phrase translation
- RBMT offers structural clarity but lacks flexibility
- Metrics: BLEU, ROUGE-L, METEOR, COMET, BERTScore

---

## 📄 Report & 📽️ Presentation

📘 **Report**: [Overleaf Report](https://www.overleaf.com/5415482843dvgjybfhpscv#72238e)  
🎞️ **Presentation**: [Powerpoint](https://husteduvn-my.sharepoint.com/:p:/g/personal/thanh_lh225458_sis_hust_edu_vn/ETnqyxmFsBhEiZ7JjPRouikBK2uBg68idFW2ULD4LlKNYw?e=1QuQPN&fbclid=IwY2xjawKgnn5leHRuA2FlbQIxMABicmlkETFabFVyRTVQTnJEMmJOdHhDAR5hLq9Rcxg81FMn_SXBi942TMgEQPHE0wFt0RD3SienVGWOx-yGR3191XeRfw_aem_M50p8C6oGJIytDMLzo0aQw)

---

## 📖 References

- [BLEU](https://aclanthology.org/P02-1040.pdf)  
- [ROUGE](https://aclanthology.org/W04-1013.pdf)  
- [METEOR](https://aclanthology.org/W05-0909.pdf)  
- [COMET](https://aclanthology.org/2020.emnlp-main.213.pdf)  
- [BERTScore](https://openreview.net/pdf?id=SkeHuCVFDr)  
- [mBART50 GitHub](https://github.com/Vu0401/NMT_mBART50-Machine-Translation)  
- [T5 Documentation](https://huggingface.co/docs/transformers/model_doc/t5)

---

> *Developed by Group 13, Hanoi University of Science and Technology – School of Information and Communication Technology*
