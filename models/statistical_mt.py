import pandas as pd
from nltk.translate import AlignedSent
from nltk.translate.ibm1 import IBMModel1
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline
from collections import defaultdict, Counter
import math
import os
from tqdm import tqdm
import pickle
import random
import gc
import matplotlib.pyplot as plt
import numpy as np
BILINGUAL_DATA_PATH = "bilingual_cleaned_dataset.csv"  # Default bilingual dataset path
VIE_DATA_PATH = "vie_cleaned_dataset.csv"  # Default Vietnamese dataset path
VISUALIZATION_PATH = "visualizations"  # Default visualization output path
BEAM_SIZE = 3 
MAX_PHRASE_LENGTH = 7  
LM_ORDER = 3  
ALPHA = 0.7
BETA = 0.3
BATCH_SIZE = 1000  # For processing data in batches
MIN_PHRASE_COUNT = 3  # Increased threshold to reduce phrase table size
LIMIT_VOCAB = 100000  # Limit vocabulary size to 10 words
MODE_VISUALIZATION = False  # Enable visualization
from pyvi import ViTokenizer
from nltk.tokenize import word_tokenize




################################################## 1. Language Model ##################################################
class LanguageModel:
    """Memory-optimized Language Model"""    
    def __init__(self, order=LM_ORDER, MODE_VISUALIZATION=MODE_VISUALIZATION):
        self.order = order
        self.lm = None
        self.vocab_size = 0
        self.MODE_VISUALIZATION = MODE_VISUALIZATION
    
    def preprocess(self, text):
        """Tokenize Vietnamese words"""
        # return text.lower().split()
        return ViTokenizer.tokenize(text.lower()).split()
    
    def visualize_iterations(self, word_freq, iteration, batch_tokens, output_dir="/kaggle/working/visualizations"):
        if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
            # Đang chạy trên Kaggle
            output_dir = "/kaggle/working/visualizations"
        else:
            output_dir = VISUALIZATION_PATH
        os.makedirs(output_dir, exist_ok=True)
        
        """Visualize word frequency for a given iteration"""
        if not self.MODE_VISUALIZATION:
            return
        
        print(f"\nIteration {iteration} - Word Frequency (Top 5):")
        top_words = word_freq.most_common(5)
        for word, count in top_words:
            print(f"  {word}: {count}")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        words, counts = zip(*word_freq.most_common(10)) if word_freq else ([], [])
        if words:
            plt.figure(figsize=(8, 6))
            plt.bar(words, counts, color='purple', alpha=0.7)
            plt.title(f'Word Frequency - Iteration {iteration}')
            plt.xlabel('Words')
            plt.ylabel('Frequency')
            plt.xticks(rotation=45)
            plt.grid(True, axis='y')
            plt.savefig(os.path.join(output_dir, f'word_freq_iter_{iteration}.png'))
            plt.close()
    
    def get_probability(self, tokens):
        """Calculate probability P(V) for a vietnamese tokens sequence"""
        if not tokens or not self.lm:
            return 0.0
        
        start_tokens = ['<s>'] * (self.order - 1)
        tokens = start_tokens + tokens
        log_prob = 0.0
        
        for i in range(self.order - 1, len(tokens)):
            context = tokens[max(0, i - self.order + 1):i]
            word = tokens[i]
            prob = self.lm.score(word, context) or 1e-10
            log_prob += math.log(prob)
        
        return log_prob
    
    def visualize_log_probabilities(self, sentences, max_sentences=100, output_dir="/kaggle/working/visualizations"):
        if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
            # Đang chạy trên Kaggle
            output_dir = "/kaggle/working/visualizations"
        else:
            # Chạy local
            output_dir = VISUALIZATION_PATH

        os.makedirs(output_dir, exist_ok=True)
        """Visualize the log probabilities of a sample of sentences"""
        if not self.MODE_VISUALIZATION:
            return
        
        if not self.lm:
            print("Cannot visualize log probabilities: Language model not trained.")
            return
        
        # Sample sentences to reduce computation
        sample_size = min(len(sentences), max_sentences)
        sample_sentences = random.sample(sentences, sample_size) if len(sentences) > max_sentences else sentences
        
        # Compute log probabilities
        log_probs = []
        for sent in sample_sentences:
            tokens = self.preprocess(sent)
            log_prob = self.get_probability(tokens)
            log_probs.append(log_prob)
        
        # Print summary statistics
        print(f"\nLog Probabilities for {len(log_probs)} sentences:")
        print(f"  Mean Log Probability: {np.mean(log_probs):.2f}")
        print(f"  Min Log Probability: {min(log_probs):.2f}")
        print(f"  Max Log Probability: {max(log_probs):.2f}")
        
        # Plot histogram of log probabilities
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        plt.figure(figsize=(8, 6))
        plt.hist(log_probs, bins=30, color='blue', alpha=0.7)
        plt.title('Distribution of Log Probabilities for Sentences')
        plt.xlabel('Log Probability')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'log_probabilities.png'))
        plt.close()
        print(f"Log probabilities visualization saved to {output_dir}/log_probabilities.png")
    
    def train(self, vietnamese_sentences, max_sentences=200000):
        """Training Language Model with memory optimization"""
        print(f"Training Language Model on {min(len(vietnamese_sentences), max_sentences)} sentences...")
        
        # Limit training data for LM to reduce memory
        if len(vietnamese_sentences) > max_sentences:
            print(f"Sampling {max_sentences} sentences from {len(vietnamese_sentences)} for LM training")
            vietnamese_sentences = random.sample(vietnamese_sentences, max_sentences)
        
        # Process in batches to reduce memory usage
        all_tokens = []
        batch_size = 10000
        word_freq = Counter()
        iteration = 0
        
        for i in range(0, len(vietnamese_sentences), batch_size):
            batch = vietnamese_sentences[i:i+batch_size]
            batch_tokens = [self.preprocess(sent) for sent in batch]
            all_tokens.extend(batch_tokens)
            
            # Update word frequency for visualization
            if self.MODE_VISUALIZATION and iteration < 2:  # Limit to 2 iterations
                for tokens in batch_tokens:
                    word_freq.update(tokens)
                self.visualize_iterations(word_freq, iteration + 1, batch_tokens)
                iteration += 1
            
            # Force garbage collection
            if i % (batch_size * 5) == 0:
                gc.collect()
        
        vocab = set()
        for tokens in all_tokens:
            vocab.update(tokens)
        
        # Limit vocabulary size to most frequent words
        if len(vocab) > LIMIT_VOCAB:
            word_freq = Counter()
            for tokens in all_tokens:
                word_freq.update(tokens)
            
            # Keep only top words
            most_common = word_freq.most_common(LIMIT_VOCAB)
            vocab = set(word for word, _ in most_common)
            print(f"Limited vocabulary to {len(vocab)} most frequent words")
        
        self.vocab_size = len(vocab)
        
        # Filter sentences to contain only vocabulary words
        filtered_sentences = []
        for tokens in all_tokens:
            filtered_tokens = [token for token in tokens if token in vocab]
            if filtered_tokens:  # Only add non-empty sentences
                filtered_sentences.append(filtered_tokens)
        
        # Clear original data
        del all_tokens
        gc.collect()
        
        # Train N-gram model
        train_data, padded_sents = padded_everygram_pipeline(self.order, filtered_sentences)
        self.lm = MLE(self.order)
        self.lm.fit(train_data, padded_sents)
        
        # Visualize log probabilities after training
        if self.MODE_VISUALIZATION:
            self.visualize_log_probabilities(vietnamese_sentences)
        
        # Clear training data
        del filtered_sentences, train_data, padded_sents
        gc.collect()
        
        return {"vocab_size": self.vocab_size, "ngram_order": self.order}

############################################# 2. Translation Model #############################################

class TranslationModel:
    """Memory-optimized Translation Model"""
    def __init__(self, max_phrase_length=MAX_PHRASE_LENGTH, MODE_VISUALIZATION=MODE_VISUALIZATION):
        self.max_phrase_length = max_phrase_length
        self.phrase_table = {}
        self.word_alignments = []
        self.MODE_VISUALIZATION = MODE_VISUALIZATION
    
    def preprocess(self, text, lang):
        """Preprocess text for both languages"""
        text = text.lower()
        if lang == 'eng':
            return word_tokenize(text)
        elif lang == 'vie':
            return ViTokenizer.tokenize(text).split()
        else:
            return text.split()
    
    def load_bilingual_data_batch(self, file_path, batch_size=BATCH_SIZE):
        """Load bilingual data in batches to reduce memory usage"""
        print(f"Loading bilingual data from {file_path} in batches")
        # default = '/kaggle/input/general-data/bilingual_cleaned_dataset.csv'
        try:
            df = pd.read_csv(file_path)
        except FileNotFoundError:
            file_path = os.path.join('datatest', BILINGUAL_DATA_PATH)
            df = pd.read_csv(file_path)
        total_rows = len(df)
        print(f"Total rows: {total_rows}")

        for start_idx in range(0, total_rows, batch_size):
            end_idx = min(start_idx + batch_size, total_rows)
            batch_df = df.iloc[start_idx:end_idx]
            
            aligned_sentences = []
            for _, row in batch_df.iterrows():
                eng_tokens = self.preprocess(row['en'], 'eng')
                vie_tokens = self.preprocess(row['vi'], 'vie')
                
                # Filter out very long sentences to save memory
                if len(eng_tokens) <= 50 and len(vie_tokens) <= 50:
                    aligned_sentences.append(AlignedSent(eng_tokens, vie_tokens))
            
            yield aligned_sentences
            
            # Clean up batch
            del batch_df, aligned_sentences
            gc.collect()
            
    def visualize_alignments(self, aligned_sentences, max_sentences=2, output_dir="/kaggle/working/visualizations"):
        if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
            # Đang chạy trên Kaggle
            output_dir = "/kaggle/working/visualizations"
        else:
            # Chạy local
            output_dir = VISUALIZATION_PATH

        os.makedirs(output_dir, exist_ok=True)
        """Visualize word alignments for a sample of sentence pairs"""
        if not self.MODE_VISUALIZATION:
            return
        
        if not self.ibm_model:
            print("Cannot visualize alignments: IBM Model 1 not trained.")
            return
        
        # Sample sentences to reduce computation
        sample_size = min(len(aligned_sentences), max_sentences)
        sample_sentences = random.sample(aligned_sentences, sample_size) if len(aligned_sentences) > max_sentences else aligned_sentences
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        for idx, sent in enumerate(sample_sentences):
            src_words = sent.words  # English
            tgt_words = sent.mots   # Vietnamese
            alignment = sent.alignment
            
            # Create alignment matrix
            matrix = np.zeros((len(tgt_words), len(src_words)))
            for src_idx, tgt_idx in alignment:
                if tgt_idx is not None and src_idx < len(src_words) and tgt_idx < len(tgt_words):
                    matrix[tgt_idx, src_idx] = 1
            
            # Plot alignment matrix
            plt.figure(figsize=(8, 6))
            plt.imshow(matrix, cmap='Blues', interpolation='nearest')
            plt.title(f'Alignment Matrix - Sentence Pair {idx + 1}')
            plt.xlabel('English Words')
            plt.ylabel('Vietnamese Words')
            plt.xticks(range(len(src_words)), src_words, rotation=45, ha='right')
            plt.yticks(range(len(tgt_words)), tgt_words)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'alignment_matrix_{idx + 1}.png'))
            plt.close()
            
            # Print alignment details
            print(f"\nSentence Pair {idx + 1}:")
            print(f"  English: {' '.join(src_words)}")
            print(f"  Vietnamese: {' '.join(tgt_words)}")
            print(f"  Alignments: {[(src_words[src], tgt_words[tgt]) for src, tgt in alignment if tgt is not None]}")
        
        print(f"Alignment visualizations saved to {output_dir}/")
        
    def _extract_alignments_memory_efficient(self, aligned_sentences, ibm_model):
        """Memory-efficient alignment extraction"""
        alignments = []
        
        # Process in smaller batches
        batch_size = 5000
        for i in range(0, len(aligned_sentences), batch_size):
            batch_alignments = []
            batch_sentences = aligned_sentences[i:i+batch_size]
            
            for sent_pair in batch_sentences:
                eng_tokens = sent_pair.words
                vie_tokens = sent_pair.mots
                
                # Only keep high-probability alignments
                alignment = []
                for eng_i, eng_word in enumerate(eng_tokens):
                    best_prob = 0
                    best_vie_i = -1
                    
                    for vie_i, vie_word in enumerate(vie_tokens):
                        prob = ibm_model.translation_table.get(eng_word, {}).get(vie_word, 0)
                        if prob > best_prob:
                            best_prob = prob
                            best_vie_i = vie_i
                    
                    # Only keep alignments above threshold
                    if best_prob > 0.01:  # Increased threshold
                        alignment.append((eng_i, best_vie_i))
                
                batch_alignments.append(alignment)
            
            alignments.extend(batch_alignments)
            
            # Periodic cleanup
            if i % (batch_size * 10) == 0:
                gc.collect()
        
        return alignments
    
    def extract_phrases_memory_efficient(self, aligned_sentences):
        """Memory-efficient phrase extraction"""
        print("Extracting phrase pairs with memory optimization...")
        
        # Use smaller data structures
        phrase_counts = defaultdict(lambda: defaultdict(int))
        
        # Process in batches
        batch_size = 5000
        for i in range(0, len(aligned_sentences), batch_size):
            batch_sentences = aligned_sentences[i:i+batch_size]
            batch_alignments = self.word_alignments[i:i+batch_size]
            
            for sent_pair, alignments in zip(batch_sentences, batch_alignments):
                if not alignments:  # Skip sentences with no alignments
                    continue
                    
                eng_tokens = sent_pair.words
                vie_tokens = sent_pair.mots
                alignment_set = set(alignments)
                
                # Extract word-level translations first
                for eng_i, vie_i in alignments:
                    if eng_i < len(eng_tokens) and vie_i < len(vie_tokens):
                        eng_word = eng_tokens[eng_i]
                        vie_word = vie_tokens[vie_i]
                        phrase_counts[eng_word][vie_word] += 1
                
                # Extract short phrases only (max length 3 to save memory)
                max_len = min(3, self.max_phrase_length)
                consistent_phrases = self._extract_consistent_phrases(
                    eng_tokens, vie_tokens, alignment_set, max_len
                )
                
                for eng_phrase, vie_phrase in consistent_phrases:
                    phrase_counts[eng_phrase][vie_phrase] += 1
            
            # Periodic cleanup
            if i % (batch_size * 5) == 0:
                gc.collect()
                print(f"Processed {i+batch_size} sentences...")
        
        # Calculate probabilities with higher threshold
        self.phrase_table = {}
        for eng_phrase, vie_phrases in phrase_counts.items():
            total_count = sum(vie_phrases.values())
            if total_count >= MIN_PHRASE_COUNT:  # Higher threshold
                # Keep only top 3 translations per phrase to save memory
                sorted_phrases = sorted(vie_phrases.items(), key=lambda x: x[1], reverse=True)[:3]
                
                filtered_phrases = {}
                for vie_phrase, count in sorted_phrases:
                    if count >= MIN_PHRASE_COUNT:
                        filtered_phrases[vie_phrase] = count / total_count
                
                if filtered_phrases:
                    self.phrase_table[eng_phrase] = filtered_phrases
        
        print(f"Extracted {len(self.phrase_table)} phrase pairs (filtered)")
        # Visualize phrase table if enabled
        if self.MODE_VISUALIZATION:
            self.visualize_phrase_table()
        
        return self.phrase_table
    
    def _extract_consistent_phrases(self, eng_tokens, vie_tokens, alignments, max_length):
        """Extract consistent phrase pairs with length limit"""
        consistent_phrases = []
        eng_len = len(eng_tokens)
        
        # Limit phrase extraction to reduce memory
        for e_start in range(eng_len):
            for e_end in range(e_start, min(eng_len, e_start + max_length)):
                vie_positions = set()
                for e_pos in range(e_start, e_end + 1):
                    for (eng_idx, vie_idx) in alignments:
                        if eng_idx == e_pos:
                            vie_positions.add(vie_idx)
                
                if not vie_positions:
                    continue
                
                v_start, v_end = min(vie_positions), max(vie_positions)
                
                if v_end - v_start + 1 <= max_length:
                    if self._is_consistent_phrase_pair(e_start, e_end, v_start, v_end, alignments):
                        eng_phrase = ' '.join(eng_tokens[e_start:e_end+1])
                        vie_phrase = ' '.join(vie_tokens[v_start:v_end+1])
                        consistent_phrases.append((eng_phrase, vie_phrase))
        
        return consistent_phrases
    
    def _is_consistent_phrase_pair(self, e_start, e_end, v_start, v_end, alignments):
        """Check if a phrase pair is consistent"""
        for (eng_idx, vie_idx) in alignments:
            if (e_start <= eng_idx <= e_end) and not (v_start <= vie_idx <= v_end):
                return False
            if (v_start <= vie_idx <= v_end) and not (e_start <= eng_idx <= e_end):
                return False
        return True

    def train_ibm_model_incremental(self, file_path="/kaggle/input/general-data/bilingual_cleaned_dataset.csv", iterations=5):
        """Train IBM Model 1 incrementally to reduce memory usage"""
        if not os.path.exists(file_path):
            file_path = os.path.join('datatest', BILINGUAL_DATA_PATH)
        print(f"Training IBM Model 1 incrementally with {iterations} iterations...")
        
        # First pass: collect vocabulary and create aligned sentences
        all_aligned_sentences = []
        eng_vocab = set()
        vie_vocab = set()
        
        for batch in self.load_bilingual_data_batch(file_path):
            for sent_pair in batch:
                eng_vocab.update(sent_pair.words)
                vie_vocab.update(sent_pair.mots)
                all_aligned_sentences.append(sent_pair)
            
            # Limit total sentences to prevent memory issues
            if len(all_aligned_sentences) >= 300000:  # Reduced from 500k
                print(f"Limited training to {len(all_aligned_sentences)} sentences")
                break
        
        print(f"Training on {len(all_aligned_sentences)} aligned sentences")
        print(f"English vocab: {len(eng_vocab)}, Vietnamese vocab: {len(vie_vocab)}")
        
        ibm_model = IBMModel1(all_aligned_sentences, iterations)
        
        # Extract alignments with memory optimization
        self.word_alignments = self._extract_alignments_memory_efficient(all_aligned_sentences, ibm_model)
        
        # Clean up
        del ibm_model
        gc.collect()
        
        return all_aligned_sentences
    
    def visualize_phrase_table(self, max_phrases=10, output_dir="/kaggle/working/visualizations"):
        if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
            # Đang chạy trên Kaggle
            output_dir = "/kaggle/working/visualizations"
        else:
            # Chạy local
            output_dir = VISUALIZATION_PATH

        os.makedirs(output_dir, exist_ok=True)
        """Visualize the phrase table as a heatmap with English phrases as columns and Vietnamese phrases as rows"""
        if not self.MODE_VISUALIZATION:
            return
        
        if not self.phrase_table:
            print("Cannot visualize phrase table: Phrase table is empty.")
            return
        
        # Select top English phrases and their top Vietnamese translations
        eng_phrases = sorted(self.phrase_table.keys(), key=lambda x: sum(self.phrase_table[x].values()), reverse=True)[:max_phrases]
        vie_phrases = set()
        for eng in eng_phrases:
            vie_phrases.update(self.phrase_table[eng].keys())
        vie_phrases = sorted(list(vie_phrases))[:max_phrases]  # Limit Vietnamese phrases
        
        # Create matrix for probabilities
        matrix = np.zeros((len(vie_phrases), len(eng_phrases)))
        for i, vie in enumerate(vie_phrases):
            for j, eng in enumerate(eng_phrases):
                matrix[i, j] = self.phrase_table.get(eng, {}).get(vie, 0)
        
        # Create heatmap
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(matrix, cmap='Blues', interpolation='nearest')
        plt.title('Phrase Table Translation Probabilities')
        plt.xlabel('English Phrases')
        plt.ylabel('Vietnamese Phrases')
        plt.xticks(range(len(eng_phrases)), eng_phrases, rotation=45, ha='right')
        plt.yticks(range(len(vie_phrases)), vie_phrases)
        plt.colorbar(label='Translation Probability')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'phrase_table.png'))
        plt.close()
        
        # Print sample phrase pairs
        print("\nSample Phrase Table Entries (Top 5 English phrases):")
        for eng in eng_phrases[:5]:
            print(f"  English: {eng}")
            for vie, prob in sorted(self.phrase_table[eng].items(), key=lambda x: x[1], reverse=True)[:3]:
                print(f"    -> Vietnamese: {vie}, Probability: {prob:.4f}")
        
        print(f"Phrase table visualization saved to {output_dir}/phrase_table.png")

############################################# 3. Decoder Algorithm #############################################

class Decoder:
    """Memory-optimized decoder"""
    def __init__(self, phrase_table, language_model, beam_size=BEAM_SIZE):
        self.phrase_table = phrase_table
        self.lm = language_model
        self.beam_size = beam_size
    def translate(self, sentence):
        """Translate sentence with memory optimization"""
        tokens = sentence.lower().split()
        if not tokens:
            return ""
        return self._greedy_translate(tokens)
        
    def _greedy_translate(self, tokens):
        """Greedy translation to save memory"""
        translation = []
        i = 0
        
        while i < len(tokens):
            best_phrase_len = 1
            best_translation = tokens[i]  # fallback
            
            # Try phrases of different lengths
            for phrase_len in range(min(3, len(tokens) - i), 0, -1):  # Max length 3
                eng_phrase = ' '.join(tokens[i:i+phrase_len])
                
                if eng_phrase in self.phrase_table:
                    # Get best translation
                    vie_translations = self.phrase_table[eng_phrase]
                    if vie_translations:
                        best_vie_phrase = max(vie_translations.items(), key=lambda x: x[1])
                        best_translation = best_vie_phrase[0]
                        best_phrase_len = phrase_len
                        break
            
            translation.append(best_translation)
            i += best_phrase_len
        
        return ' '.join(translation)
    
class Hypothesis:
    """Lightweight hypothesis class"""
    def __init__(self, translation, coverage, score, last_phrase_end):
        self.translation = translation
        self.coverage = coverage
        self.score = score
        self.last_phrase_end = last_phrase_end

################################################# 4. Combine all SMT System #############################################
class SMT:
    """Memory-optimized SMT system"""
    def __init__(self):
        self.lm = LanguageModel(order=LM_ORDER)
        self.tm = TranslationModel(max_phrase_length=MAX_PHRASE_LENGTH)
        self.decoder = None
        
    def post_process(self, text):
        """Replaces underscores with spaces in the translated text."""
        return text.replace("_", " ")
    
    def train(self):
        bilingual_path = "/kaggle/input/general-data/bilingual_cleaned_dataset.csv"
        vie_path = "/kaggle/input/general-data/vie_cleaned_dataset.csv"
        
        if not os.path.exists(bilingual_path):
            bilingual_path = os.path.join("datatest", BILINGUAL_DATA_PATH)
            vie_path = os.path.join("datatest", VIE_DATA_PATH)
        print("=== Training Translation Model ===")
        aligned_sentences = self.tm.train_ibm_model_incremental(bilingual_path)
        phrase_table = self.tm.extract_phrases_memory_efficient(aligned_sentences)
        
        del aligned_sentences
        gc.collect()
        
        # Train language model
        print("\n=== Training Language Model ===")
        vie_df = pd.read_csv(vie_path)
        vietnamese_sentences = vie_df['vi'].tolist()
        del vie_df  # Free memory
        gc.collect()
        
        lm_stats = self.lm.train(vietnamese_sentences, max_sentences=50000)  # Limit LM training data
        del vietnamese_sentences  # Free memory
        gc.collect()
        
        # Initialize decoder
        self.decoder = Decoder(phrase_table, self.lm)
        
        # Save model immediately
        self.save_model()
        
        return {
            "phrase_pairs": len(phrase_table),
            "lm_stats": lm_stats
        }
    
    def translate_sentence(self, sentence):
        """Translate a single sentence"""
        if self.decoder is None:
            raise ValueError("Model not trained or loaded.")
        translated_text_with_underscores = self.decoder.translate(sentence)
        return self.post_process(translated_text_with_underscores)
    
    def save_model(self):
        """Save the trained model"""
        if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
            # Đang chạy trên Kaggle
            model_dir = "/kaggle/working/checkpoints"
        else:
            # Chạy local
            model_dir = "checkpoints"

        os.makedirs(model_dir, exist_ok=True)
        
        # Save with compression
        with open(os.path.join(model_dir, "phrase_table.pkl"), 'wb') as f:
            pickle.dump(self.tm.phrase_table, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(model_dir, "lm_object.pkl"), 'wb') as f:
            pickle.dump(self.lm, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"Model saved to {model_dir}")
        
    def load_model(self, model_dir='checkpoints'):
        """Load a pre-trained model"""
        with open(os.path.join(model_dir, "phrase_table.pkl"), 'rb') as f:
            phrase_table = pickle.load(f)
        with open(os.path.join(model_dir, "lm_object.pkl"), 'rb') as f:
            self.lm = pickle.load(f)
        
        self.decoder = Decoder(phrase_table, self.lm, BEAM_SIZE)
        self.tm.phrase_table = phrase_table
        
        print(f"Model loaded from {model_dir}")
    
    def evaluate(self, test_file='/kaggle/input/general-data/test_cleaned_dataset.csv', sample_size=5):
        """Evaluate model on test set"""
        try :
            df = pd.read_csv(test_file) 
        except FileNotFoundError:   
            test_file = 'datatest/test_cleaned_dataset.csv'
            df = pd.read_csv(test_file)
        sample_size = min(sample_size, len(df))
        sample_indices = random.sample(range(len(df)), sample_size)
        
        results = []
        for idx in sample_indices:
            try:
                source = df.iloc[idx]['en']
                reference = df.iloc[idx]['vi']
                translation = self.translate_sentence(source)
                
                results.append({
                    "source": source,
                    "reference": reference,
                    "translation": translation
                })
            except Exception as e:
                print(f"Error translating sentence {idx}: {e}")
                results.append({
                    "source": df.iloc[idx]['en'],
                    "reference": df.iloc[idx]['vi'],
                    "translation": "Translation failed"
                })
        
        return results    
    
    def save_predictions_batch(self, test_file="/kaggle/input/general-data/test_cleaned_dataset.csv", output_file="/kaggle/working/predicted.csv", batch_size=1000):
        """Save predictions in batches to avoid memory issues"""
        # Check if test_file exists, if not update to default path
        if not os.path.exists(test_file):
            test_file = "datatest/test_cleaned_dataset.csv"
            output_file = "datatest/predicted1.csv"
        print(f"Output file will be saved to: {output_file}")
        
        df_info = pd.read_csv(test_file, nrows=0)  # Just get column info
        total_rows = len(pd.read_csv(test_file))
        
        print(f"Processing {total_rows} sentences in batches of {batch_size}")
        
        # Process in batches and write incrementally
        first_batch = True
        for start_idx in tqdm(range(0, total_rows, batch_size), desc="Processing batches"):
            end_idx = min(start_idx + batch_size, total_rows)
            
            # Read batch
            batch_df = pd.read_csv(test_file, skiprows=range(1, start_idx+1), nrows=batch_size)
            
            # Process batch
            batch_predictions = []
            for _, row in batch_df.iterrows():
                try:
                    source = row['en']
                    reference = row['vi']
                    translation = self.translate_sentence(source)
                    
                    batch_predictions.append({
                        "en": source,
                        "vi": reference,
                        "pre": translation
                    })
                except Exception as e:
                    batch_predictions.append({
                        "en": row['en'],
                        "vi": row['vi'],
                        "pre": "Translation failed"
                    })
            
            # Save batch
            batch_pred_df = pd.DataFrame(batch_predictions)
            
            if first_batch:
                batch_pred_df.to_csv(output_file, index=False)
                first_batch = False
            else:
                batch_pred_df.to_csv(output_file, mode='a', header=False, index=False)
            
            # Clean up
            del batch_df, batch_predictions, batch_pred_df
            gc.collect()
        
        print(f"Predictions saved to {output_file}")
        return output_file
    
def main():
    print("Starting Memory-Optimized SMT System...")
    smt = SMT()
    model_dir = "checkpoints"
    if os.path.exists(model_dir) and os.path.isfile(os.path.join(model_dir, "phrase_table.pkl")):
        print("Loading existing model...")
        smt.load_model()
    else:
        print("Training new model...")
        stats = smt.train()
        print(f"Training complete: {stats}")
        
    # Evaluate model
    print("\nEvaluating model...")
    results = smt.evaluate(sample_size=1)  
    print("\nExample translations:")
    for i, result in enumerate(results):
        print(f"\nExample {i+1}:")
        print(f"English:    {result['source']}")
        print(f"Reference:  {result['reference']}")
        print(f"Translation: {result['translation']}")
        
    # Save predictions in batches
    print("\nSaving predictions in batches...")
    output_file = smt.save_predictions_batch(batch_size=500)  # Smaller batch size
    print(f"All predictions saved to: {output_file}")
    
    # Final memory cleanup
    gc.collect()
    print("Processing complete!")

class SMTExtended(SMT):
    def infer(self, sentence):
        """Translate a single arbitrary English sentence into Vietnamese using beam search"""
        if self.decoder is None:
            raise ValueError("Model not trained or loaded.")
        
        # Preprocess input sentence
        tokens = self.tm.preprocess(sentence, 'eng')
        if not tokens:
            return ""
        
        # Initialize beam: (score, translation_tokens, last_pos, covered_positions)
        beam = [(0.0, [], 0, set())]  # Score, translation tokens, last position, covered positions
        best_score = float('-inf')
        best_translation = []
        
        # Beam search
        while beam:
            new_beam = []
            for score, trans_tokens, last_pos, covered in beam:
                # Check if all positions are covered
                if len(covered) == len(tokens):
                    if score > best_score:
                        best_score = score
                        best_translation = trans_tokens
                    continue
                
                # Find next uncovered position
                next_pos = last_pos
                while next_pos in covered and next_pos < len(tokens):
                    next_pos += 1
                
                if next_pos >= len(tokens):
                    if score > best_score:
                        best_score = score
                        best_translation = trans_tokens
                    continue
                
                # Try phrases starting at next_pos
                for phrase_len in range(1, min(self.tm.max_phrase_length + 1, len(tokens) - next_pos + 1)):
                    eng_phrase = ' '.join(tokens[next_pos:next_pos + phrase_len])
                    
                    # Get possible translations from phrase table
                    vie_translations = self.tm.phrase_table.get(eng_phrase, {})
                    if not vie_translations and phrase_len == 1:
                        # Fallback for single unknown word
                        vie_translations = {tokens[next_pos]: 1.0}
                    
                    for vie_phrase, trans_prob in vie_translations.items():
                        # Split Vietnamese phrase into tokens for LM scoring
                        vie_tokens = vie_phrase.split()
                        # Calculate new score: combine translation prob and LM prob
                        log_trans_prob = math.log(trans_prob) if trans_prob > 0 else math.log(1e-10)
                        lm_score = self.lm.get_probability(trans_tokens + vie_tokens)
                        new_score = ALPHA * log_trans_prob + BETA * lm_score
                        
                        # Update covered positions
                        new_covered = covered | set(range(next_pos, next_pos + phrase_len))
                        # Add to new beam
                        new_beam.append((score + new_score, trans_tokens + vie_tokens, next_pos + phrase_len, new_covered))
            
            # Keep top BEAM_SIZE hypotheses
            new_beam.sort(key=lambda x: x[0], reverse=True)
            beam = new_beam[:self.decoder.beam_size]
        
        # Return best translation
        return ' '.join(best_translation) if best_translation else "Translation failed"

if __name__ == "__main__":
    main()
    
