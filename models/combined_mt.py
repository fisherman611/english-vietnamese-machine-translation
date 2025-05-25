import os
import sys 
import pandas as pd 
import numpy as np 
import pickle
import gc 
from collections import defaultdict, Counter 
import nltk 
from nltk.translate import IBMModel1, AlignedSent
from nltk.lm import MLE 
from nltk.lm.preprocessing import padded_everygram_pipeline
import math 
import random
from tqdm import tqdm 
import matplotlib.pyplot as plt 

from rule_based_mt import TransferBasedMT
from statistical_mt import SMT, LanguageModel, TranslationModel, Decoder

class PostEditingModel:
    """
    Statistical Post-Editing Model that takes RBMT output and improves it
    using phrase-based statistical machine translation
    """
    def __init__(self, order=3, max_phrase_length=5, beam_size=3):
        self.order = order
        self.max_phrase_length = max_phrase_length
        self.beam_size = beam_size
        
        # Post-editing specific components
        self.pe_phrase_table = {}
        self.pe_language_model = None
        self.pe_decoder = None
        
        # Training data will be: (RBMT_output, Reference_translation) pairs
        self.rbmt_outputs = []
        self.reference_translations = []
    

    def prepare_post_editing_data(self, bilingual_data_path, rbmt_translator):
        """
        Generate post-editing training data by running RBMT on source sentences
        and pairing outputs with reference translations
        """
        print("Preparing post-editing training data...")
        
        try:
            df = pd.read_csv(bilingual_data_path)
        except FileNotFoundError:
            bilingual_data_path = os.path.join('datatest', 'bilingual_lor.csv')
            df = pd.read_csv(bilingual_data_path)
        
        # Limit data size for memory efficiency
        max_samples = min(len(df), 10000)  # Reduced for demonstration
        if len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=42)
        
        pe_data = []
        print(f"Processing {len(df)} sentences for post-editing data...")
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating RBMT outputs"):
            try:
                english_sentence = row['en']
                reference_vietnamese = row['vi']
                
                # Get RBMT translation (suppress output)
                original_stdout = sys.stdout
                sys.stdout = open(os.devnull, 'w')
                rbmt_output = rbmt_translator.translate(english_sentence)
                sys.stdout.close()
                sys.stdout = original_stdout
                
                if rbmt_output and reference_vietnamese:
                    pe_data.append({
                        'rbmt_output': rbmt_output,
                        'reference': reference_vietnamese,
                        'source': english_sentence
                    })
                    
            except Exception as e:
                print(f"Error processing sentence {idx}: {e}")
                continue
        
        print(f"Generated {len(pe_data)} post-editing training pairs")
        return pe_data
    
    
    def train_post_editing_model(self, pe_data):
        """
        Train the post-editing model using RBMT outputs and reference translations
        """
        print("Training post-editing model...")
        
        # Prepare aligned sentences for IBM Model training
        aligned_sentences = []
        vietnamese_sentences = []
        
        for item in pe_data:
            rbmt_tokens = item['rbmt_output'].lower().split()
            ref_tokens = item['reference'].lower().split()
            
            if len(rbmt_tokens) > 0 and len(ref_tokens) > 0:
                # Create alignment: RBMT output -> Reference translation
                aligned_sentences.append(AlignedSent(rbmt_tokens, ref_tokens))
                vietnamese_sentences.append(item['reference'])
        
        print(f"Training on {len(aligned_sentences)} aligned sentence pairs")
        
        # Train IBM Model for post-editing
        if len(aligned_sentences) > 0:
            ibm_model = IBMModel1(aligned_sentences, 5)
            
            # Extract phrase table for post-editing
            self.pe_phrase_table = self._extract_pe_phrases(aligned_sentences, ibm_model)
            
            # Train language model on reference translations
            self._train_pe_language_model(vietnamese_sentences)
            
            # Initialize post-editing decoder
            self.pe_decoder = Decoder(self.pe_phrase_table, self.pe_language_model, self.beam_size)
            
            print(f"Post-editing model trained with {len(self.pe_phrase_table)} phrase pairs")
            return True
        else:
            print("No valid training data for post-editing model")
            return False
        
    
    def _extract_pe_phrases(self, aligned_sentences, ibm_model):
        """
        Extract phrases for post-editing (RBMT output -> improved translation)
        """
        phrase_counts = defaultdict(lambda: defaultdict(int))
        
        for sent_pair in aligned_sentences:
            rbmt_tokens = sent_pair.words  # RBMT output
            ref_tokens = sent_pair.mots    # Reference translation
            
            # Extract word-level corrections
            for i, rbmt_word in enumerate(rbmt_tokens):
                best_prob = 0
                best_ref_word = rbmt_word  # fallback
                
                for ref_word in ref_tokens:
                    prob = ibm_model.translation_table.get(rbmt_word, {}).get(ref_word, 0)
                    if prob > best_prob:
                        best_prob = prob
                        best_ref_word = ref_word
                
                if best_prob > 0.01:  # Only keep confident corrections
                    phrase_counts[rbmt_word][best_ref_word] += 1
            
            # Extract short phrase corrections (length 2-3)
            for rbmt_start in range(len(rbmt_tokens)):
                for rbmt_end in range(rbmt_start, min(rbmt_start + 3, len(rbmt_tokens))):
                    rbmt_phrase = ' '.join(rbmt_tokens[rbmt_start:rbmt_end + 1])
                    
                    # Find corresponding reference phrase (simplified heuristic)
                    for ref_start in range(len(ref_tokens)):
                        for ref_end in range(ref_start, min(ref_start + 3, len(ref_tokens))):
                            ref_phrase = ' '.join(ref_tokens[ref_start:ref_end + 1])
                            
                            # Simple scoring based on word overlap
                            if self._phrases_related(rbmt_phrase, ref_phrase, ibm_model):
                                phrase_counts[rbmt_phrase][ref_phrase] += 1
        
        # Convert counts to probabilities
        phrase_table = {}
        for rbmt_phrase, ref_phrases in phrase_counts.items():
            total_count = sum(ref_phrases.values())
            if total_count >= 2:  # Minimum frequency threshold
                phrase_table[rbmt_phrase] = {}
                for ref_phrase, count in ref_phrases.items():
                    phrase_table[rbmt_phrase][ref_phrase] = count / total_count
        
        return phrase_table
    
    
    def _phrases_related(self, rbmt_phrase, ref_phrase, ibm_model):
        """
        Heuristic to determine if RBMT phrase and reference phrase are related
        """
        rbmt_words = rbmt_phrase.split()
        ref_words = ref_phrase.split()
        
        if len(rbmt_words) != len(ref_words):
            return False
        
        # Check if at least half the words have translation relationship
        related_count = 0
        for rbmt_word in rbmt_words:
            for ref_word in ref_words:
                prob = ibm_model.translation_table.get(rbmt_word, {}).get(ref_word, 0)
                if prob > 0.01:
                    related_count += 1
                    break
        
        return related_count >= len(rbmt_words) / 2
    
    
    def _train_pe_language_model(self, vietnamese_sentences):
        """
        Train language model on reference Vietnamese sentences
        """
        print("Training post-editing language model...")
        
        # Limit sentences for memory efficiency
        max_sentences = min(len(vietnamese_sentences), 50000)
        if len(vietnamese_sentences) > max_sentences:
            vietnamese_sentences = random.sample(vietnamese_sentences, max_sentences)
        
        # Tokenize sentences
        tokenized_sentences = []
        for sent in vietnamese_sentences:
            tokens = sent.lower().split()
            if len(tokens) > 0:
                tokenized_sentences.append(tokens)
        
        # Train n-gram language model
        train_data, padded_sents = padded_everygram_pipeline(self.order, tokenized_sentences)
        self.pe_language_model = MLE(self.order)
        self.pe_language_model.fit(train_data, padded_sents)
        
        print(f"Language model trained on {len(tokenized_sentences)} sentences")
        
        
    def post_edit(self, rbmt_output):
        """
        Apply post-editing to RBMT output
        """
        if not self.pe_decoder:
            return rbmt_output  # Return original if no post-editing model
        
        # Apply post-editing corrections
        corrected_output = self.pe_decoder.translate(rbmt_output)
        return corrected_output if corrected_output else rbmt_output
    
    
    def save_model(self, model_dir='pe_model'):
        """Save the post-editing model"""
        os.makedirs(model_dir, exist_ok=True)
        
        with open(os.path.join(model_dir, "pe_phrase_table.pkl"), 'wb') as f:
            pickle.dump(self.pe_phrase_table, f)
        with open(os.path.join(model_dir, "pe_language_model.pkl"), 'wb') as f:
            pickle.dump(self.pe_language_model, f)
        
        print(f"Post-editing model saved to {model_dir}")
        
    
    def load_model(self, model_dir='pe_model'):
        """Load the post-editing model"""
        with open(os.path.join(model_dir, "pe_phrase_table.pkl"), 'rb') as f:
            self.pe_phrase_table = pickle.load(f)
        with open(os.path.join(model_dir, "pe_language_model.pkl"), 'rb') as f:
            self.pe_language_model = pickle.load(f)
        
        self.pe_decoder = Decoder(self.pe_phrase_table, self.pe_language_model, self.beam_size)
        print(f"Post-editing model loaded from {model_dir}")
        

class CombinedMTSystem:
    """
    Combined Machine Translation System that uses RBMT followed by Statistical Post-Editing
    """
    def __init__(self):
        self.post_editor = PostEditingModel() 
    
    
    def train_post_editing(self, bilingual_data_path='datatest/bilingual_lor.csv'):
        """
        Train the post-editing component
        """
        print("Training post-editing component...")
        
        # Generate post-editing training data
        pe_data = self.post_editor.prepare_post_editing_data(bilingual_data_path, self.rbmt_translator)
        
        # Train post-editing model
        success = self.post_editor.train_post_editing_model(pe_data)
        
        if success:
            self.post_editor.save_model()
            print("Post-editing training completed successfully")
        else:
            print("Post-editing training failed")
        
        return success
    
    
    def load_model(self):
        """
        Load pre-trained models
        """
        # Load post-editing model if available
        if os.path.exists('pe_model') and os.path.isfile('pe_model/pe_phrase_table.pkl'):
            self.post_editor.load_model()
            print("Post-editing model loaded")
            
    
    def translate(self, english_sentence, verbose=False):
        """
        Translate using the selected mode
        """
        results = {'source': english_sentence}
        # RBMT + SMT Post-editing
        # Step 1: Get RBMT translation
        if verbose:
            rbmt_output = self.rbmt_translator.translate(english_sentence)
        else:
            original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            rbmt_output = self.rbmt_translator.translate(english_sentence)
            sys.stdout.close()
            sys.stdout = original_stdout
        
        results['rbmt'] = rbmt_output
        
        # Step 2: Apply post-editing
        post_edited = self.post_editor.post_edit(rbmt_output)
        results['post_edited'] = post_edited
        results['final'] = post_edited
    
        return results['final']

def main():
    """
    Main function to demonstrate the combined MT system
    """
    print("=== Combined RBMT + SMT Translation System ===")
    
    # Initialize the combined system
    combined_system = CombinedMTSystem()
    
    # Load existing models if available
    combined_system.load_model()

    # Test sentences
    test_sentences = [
        "I love you",
        "This is a beautiful day",
        "How are you today?",
        "I want to learn Vietnamese",
        "The weather is very nice"
    ]
    for sentence in test_sentences:
        combined_result = combined_system.translate(sentence)
        print(f"RBMT + Post-editing: {combined_result}")
        
if __name__ == "__main__":
    main()
