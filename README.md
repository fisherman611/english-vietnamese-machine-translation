# **English-Vietnamese Machine Translation**  

## 📌 Project Overview  

This project aims to develop an **English-to-Vietnamese Machine Translation System**. The goal is to explore multiple approaches—**Rule-Based Machine Translation (RBMT), Statistical Machine Translation (SMT), and Neural Machine Translation (NMT)**—to assess their effectiveness using standard evaluation metrics.  

## 🔹 **Project Scope**  

### 🔹 **Translation Approaches**  
We will implement and compare the following machine translation methods:  

- **Rule-Based Machine Translation (RBMT)** – Uses linguistic rules and grammar-based approaches.  
- **Statistical Machine Translation (SMT)** – Leverages statistical models to determine the most probable translation.  
- **Neural Machine Translation (NMT)** – Utilizes deep learning models, particularly **LSTM** and **Transformers** (pretrained models like **BERT**, **GPT**, **mBART50**, **T5**).  

### 🔹 **Evaluation Metrics**  
To assess translation quality, we will use the following metrics:  
- **BLEU** – Evaluates n-gram precision and is widely used in translation benchmarks.
- **ROUGE** – Measures recall-oriented n-gram overlap (good for longer text).
- **METEOR** – Considers stemming and synonyms for more flexible matching.
- **COMET** – Uses embeddings from a pre-trained model to evaluate the similarity between source, reference, and hypothesis translations.
- **BERTScore** – Uses transformer embeddings and computes cosine similarity between reference and hypothesis tokens.

📌 **References:**

🔗 [BLEU: a Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040.Pdf)

🔗 [ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013.pdf)

🔗 [METEOR: An Automatic Metric for MT Evaluation with Improved Correlation with Human Judgments ](https://aclanthology.org/W05-0909.pdf)

🔗 [COMET: A Neural Framework for MT Evaluation](https://aclanthology.org/2020.emnlp-main.213.pdf)

🔗 [BERTSCORE: EVALUATING TEXT GENERATION WITH BERT](https://openreview.net/pdf?id=SkeHuCVFDr)

🔗 [Exploring Robustness of Machine Translation Metrics: A Study of Twenty-Eight Automatic Metrics in the WMT22 Metric Task](https://aclanthology.org/2022.wmt-1.46.pdf)


---

## 📥 **Input**  
- An English sentence.  

## 📤 **Output**  
- The corresponding Vietnamese translation.  

---

## 📚 **Dataset**  

We will use the **English-Vietnamese parallel corpus** from Hugging Face:  
🔗 [Dataset Link](https://huggingface.co/datasets/ncduy/mt-en-vi/tree/main)  

This dataset is already divided into **three parts**:  

|       | **Train** | **Validation** | **Test** |
|-------|----------|--------------|--------|
| **Number of Examples** | 2,884,451 | 11,316 | 11,225 |

Each subset contains the following features:  
- **`en`**: English sentence.  
- **`vi`**: Corresponding Vietnamese sentence.  
- **`source`**: The source from which the example is taken.  

In this project, we focus on a randomly selected subset of the dataset, consisting of either 300,000 or 500,000 rows. Below is the specific subset used in our analysis:

🔗 [Sub-dataset Link](https://husteduvn-my.sharepoint.com/:f:/g/personal/thanh_lh225458_sis_hust_edu_vn/EnNCEp9SLQBPl2xn_qOY19QBMr_kNMVvXuK6h8JwmEBpIw?e=gar0gN)

### 🔹 **Data Processing Steps**  
✔ **Exploratory Data Analysis (EDA)**  
✔ **Data Preprocessing** (Tokenization, Cleaning, Handling Special Cases)  

---

## 🤖 **Models & Approaches**  

### **1️⃣ Rule-Based Machine Translation (RBMT)**  
- A linguistic approach using grammatical rules and dictionary lookups.  

📌 **References:**  

🔗 [Statistical Vs Rule Based Machine Translation; A Case Study on Indian Language Perspective](https://arxiv.org/pdf/1708.04559)

---

### **2️⃣ Statistical Machine Translation (SMT)**  
- Utilizes **phrase-based models**, **word alignment**, and **probability-based translation rules**.  

📌 **References:**  

🔗 [Statistical Vs Rule Based Machine Translation; A Case Study on Indian Language Perspective](https://arxiv.org/pdf/1708.04559)

---

### **3️⃣ Neural Machine Translation (NMT)**  
This approach will focus on **deep learning models** to improve translation accuracy.  

✅ **LSTM-based NMT**  
✅ **RNN-based NMT** 
✅ **Transformer-based NMT** (using pretrained models from Hugging Face)  
- **BERT**  
- **GPT**
- **mBART50**
- **T5**

📌 **References:**  

🔗 [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/pdf/1409.3215)

🔗 [Study of Neural Machine Translation With Long Short-Term Memory Techniques](https://www.researchgate.net/publication/365595688_Study-of-Neural-Machine-Translation-With-Long-Short-Term-Memory-Techniques)

🔗 [BERTTune: Fine-Tuning Neural Machine Translation with BERTScore](https://aclanthology.org/2021.acl-short.115.pdf)

🔗 [Fine-tuning Large Language Models for Adaptive Machine Translation](https://arxiv.org/pdf/2312.12740v1)

🔗 [NMT mBART50-Machine-Translation](https://github.com/Vu0401/NMT_mBART50-Machine-Translation)

🔗 [T5](https://huggingface.co/docs/transformers/model_doc/t5)

---

## 📄**Report**
🔗 [Report Link](https://www.overleaf.com/5415482843dvgjybfhpscv#72238e)

---
## 📽️ **Presentation**
🔗 [Presentation link](https://husteduvn-my.sharepoint.com/:p:/g/personal/thanh_lh225458_sis_hust_edu_vn/EZsFflPryDhKmrbz-KXZpVkBXdaCxsAYysBs5ec1Wp6UVA?e=tdTHb3)
