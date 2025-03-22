from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import numpy as np

from evaluate import load
'''Uncomment when run on Kaggle or Google Colab'''

# comet_metric = load("comet")
# bertscore_metric = load("bertscore")


class BLEU:
    """
    BLEU (Bilingual Evaluation Understudy) score calculation class.
    Computes corpus-level BLEU scores.
    """

    def __init__(self, references: list[str], candidates: list[str]) -> None:
        """
        Initializes the BLEU score calculation.

        Args:
            references (list[str]): A list of reference sentences for comparison.
            candidates (list[str]): A list of candidate (generated) sentences to evaluate.
        """
        self.references = [[reference.lower().split()]
                           for reference in references]
        self.candidates = [
            candidate.lower().split() for candidate in candidates
        ]

    def bleu(self, n=4) -> float:
        """
        Computes the BLEU score up to n-grams.

        Args:
            n (int, optional): The maximum n-gram length to consider. Defaults to 4.

        Returns:
            float: The BLEU score, ranging from 0 to 1.
        """
        return corpus_bleu(self.references,
                           self.candidates,
                           weights=(1 / n, ) * n)


class METEOR:
    """
    METEOR (Metric for Evaluation of Translation with Explicit ORdering) score calculation class.
    Computes METEOR scores for candidate sentences.
    """

    def __init__(self, references: list[str], candidates: list[str]) -> None:
        """
        Initializes the BLEU score calculation.

        Args:
            references (list[str]): A list of reference sentences for comparison.
            candidates (list[str]): A list of candidate (generated) sentences to evaluate.
        """
        self.references = [[reference.lower().split()]
                           for reference in references]
        self.candidates = [
            candidate.lower().split() for candidate in candidates
        ]

        self.count = len(self.references)

    def meteor(self) -> float:
        """
        Computes the METEOR score for the candidate sentences.

        Returns:
            float: The METEOR score, ranging from 0 to 1.
        """
        total_score = 0
        for reference, candidate in zip(self.references, self.candidates):
            total_score += meteor_score(reference, candidate)

        return total_score / self.count if self.count > 0 else 0.0


class ROUGE:
    """
    ROUGE (Recall-Oriented Understudy for Gisting Evaluation) score calculation class.
    Computes various ROUGE scores including ROUGE-N, ROUGE-L, ROUGE-W, ROUGE-S, and ROUGE-SU.
    """

    def __init__(self, references: list[str], candidates: list[str]) -> None:
        """
        Initializes the ROUGE score calculation.

        Args:
            references (list[str]): A list of reference sentences for comparison.
            candidates (list[str]): A list of candidate (generated) sentences to evaluate.
        """
        self.references = [[reference.lower()] for reference in references]
        self.candidates = [candidate.lower() for candidate in candidates]
        self.count = len(self.references)
        self.references_split = [[reference.lower().split()]
                                 for reference in references]
        self.candidates_split = [
            candidate.lower().split() for candidate in candidates
        ]

    def rouge_N(self, n=4) -> float:
        """
        Computes the ROUGE-N score based on n-gram overlap.

        Args:
            n (int, optional): The n-gram length to consider. Defaults to 4.

        Returns:
            float: The ROUGE-N score, ranging from 0 to 1.
        """
        scorer = rouge_scorer.RougeScorer([f"rouge{n}"], use_stemmer=True)

        total = 0
        for references, candidate in zip(self.references, self.candidates):
            total += sum(
                scorer.score(ref, candidate)[f"rouge{n}"].fmeasure
                for ref in references) / len(references)

        return total / self.count if self.count > 0 else 0.0

    def rouge_L(self) -> float:
        """
        Computes the ROUGE-L score based on the longest common subsequence (LCS).

        Returns:
            float: The ROUGE-L score, ranging from 0 to 1.
        """
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

        total = 0
        for references, candidate in zip(self.references, self.candidates):
            total += sum(
                scorer.score(ref, candidate)["rougeL"].fmeasure
                for ref in references) / len(references)

        return total / self.count if self.count > 0 else 0.0

    def single_rouge_W(self, reference: list[str],
                       candidate: list[str]) -> float:
        """
        Computes the ROUGE-W score for a single reference-candidate pair.

        Args:
            reference (list[str]): A single reference sentence.
            candidate (list[str]): A single candidate sentence.

        Returns:
            float: The ROUGE-W score for the pair.
        """
        m = len(reference)
        n = len(candidate)

        # Initialize the DP tables
        c = [[0] * (n + 1) for _ in range(m + 1)]
        w = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if reference[i - 1] == candidate[j - 1]:
                    k = w[i - 1][j - 1]
                    # Calculate the increment using f(k+1) - f(k) where f(k) = k^2
                    increment = (k + 1)**2 - k**2  # Simplifies to 2k + 1
                    c[i][j] = c[i - 1][j - 1] + increment
                    w[i][j] = k + 1
                else:
                    if c[i - 1][j] > c[i][j - 1]:
                        c[i][j] = c[i - 1][j]
                    else:
                        c[i][j] = c[i][j - 1]
                    w[i][j] = 0  # No consecutive match

        wlcs_score = c[m][n]

        # Compute ROUGE-W (using Recall as beta is very large)
        if m == 0:
            return 0.0  # Handle edge case to avoid division by zero
        f_inverse = lambda x: x**0.5  # Inverse of f(k) = k^2 is sqrt(k)

        rouge_score = f_inverse(wlcs_score) / m
        return rouge_score

    def rouge_W(self) -> float:
        """
        Computes the average ROUGE-W score across all reference-candidate pairs.

        Returns:
            float: The average ROUGE-W score, ranging from 0 to 1.
        """
        total = 0
        for references, candidate in zip(self.references_split,
                                         self.candidates_split):
            total += sum(
                self.single_rouge_W(ref, candidate)
                for ref in references) / len(references)

        return total / self.count if self.count > 0 else 0.0

    def get_skip_bigrams(self,
                         sentence: list[str],
                         d_skip: int = None) -> set[str]:
        """
        Generates all skip-bigrams from a sentence with an optional maximum skip distance.

        Args:
            sentence (list[str]): The sentence to generate skip-bigrams from.
            d_skip (int, optional): The maximum allowed skip distance. Defaults to None (unlimited).

        Returns:
            set[str]: A set of skip-bigrams.
        """
        bigrams = set()
        n = len(sentence)
        for i in range(n):
            # Determine valid j positions based on d_skip
            if d_skip is None:
                j_max = n
            else:
                j_max = min(i + d_skip + 2, n)  # j <= i + d_skip + 1
            for j in range(i + 1, j_max):
                bigrams.add((sentence[i], sentence[j]))
        return bigrams

    def single_rouge_S(self,
                       reference: list[str],
                       candidate: list[str],
                       d_skip: int = None,
                       beta: int = 1) -> dict:
        """
        Computes the ROUGE-S score for a single reference-candidate pair.

        Args:
            reference (list[str]): A single reference sentence.
            candidate (list[str]): A single candidate sentence.
            d_skip (int, optional): The maximum allowed skip distance. Defaults to None (unlimited).
            beta (int, optional): The weight for recall in the F-measure. Defaults to 1.

        Returns:
            dict: A dictionary containing recall, precision, and F-measure scores.
        """
        # Generate skip-bigrams for both sequences
        ref_bigrams = self.get_skip_bigrams(reference, d_skip)
        cand_bigrams = self.get_skip_bigrams(candidate, d_skip)

        # Calculate intersection (matching skip-bigrams)
        overlap = len(ref_bigrams & cand_bigrams)

        # Calculate denominator values
        ref_count = len(ref_bigrams)
        cand_count = len(cand_bigrams)

        # Calculate recall and precision (handle division by zero)
        recall = overlap / ref_count if ref_count > 0 else 0.0
        precision = overlap / cand_count if cand_count > 0 else 0.0

        # Calculate F-measure
        if recall + precision == 0:
            f_score = 0.0
        else:
            f_score = ((1 + beta**2) * recall *
                       precision) / (recall + beta**2 * precision)

        return {'recall': recall, 'precision': precision, 'f_score': f_score}

    def rouge_S(self) -> float:
        """
        Computes the average ROUGE-S score across all reference-candidate pairs.

        Returns:
            float: The average ROUGE-S score, ranging from 0 to 1.
        """
        total = 0
        for references, candidate in zip(self.references_split,
                                         self.candidates_split):
            total += sum(
                self.single_rouge_S(ref, candidate)["f_score"]
                for ref in references) / len(references)

        return total / self.count if self.count > 0 else 0.0

    def single_rouge_SU(self,
                        reference: list[str],
                        candidate: list[str],
                        d_skip: int = None,
                        beta: int = 1) -> float:
        """
        Computes the ROUGE-SU score for a single reference-candidate pair by adding a BOS marker.

        Args:
            reference (list[str]): A single reference sentence.
            candidate (list[str]): A single candidate sentence.
            d_skip (int, optional): The maximum allowed skip distance. Defaults to None (unlimited).
            beta (int, optional): The weight for recall in the F-measure. Defaults to 1.

        Returns:
            float: The ROUGE-SU score for the pair.
        """
        # Add begin-of-sentence markers
        bos = '<BOS>'
        ref_with_bos = [bos] + reference
        cand_with_bos = [bos] + candidate

        # Calculate ROUGE-S with the augmented sequences
        return self.single_rouge_S(ref_with_bos, cand_with_bos, d_skip, beta)

    def rouge_SU(self) -> float:
        """
        Computes the average ROUGE-SU score across all reference-candidate pairs.

        Returns:
            float: The average ROUGE-SU score, ranging from 0 to 1.
        """
        total = 0
        for references, candidate in zip(self.references_split,
                                         self.candidates_split):
            total += sum(
                self.single_rouge_SU(ref, candidate)["f_score"]
                for ref in references) / len(references)

        return total / self.count if self.count > 0 else 0.0


'''Uncomment when run on Kaggle or Google Colab'''
# class COMET:

#     def __init__(self, sources: list[str], references: list[str],
#                  candidates: list[str]) -> None:
#         """
#         Initializes the COMET score calculation.

#         Args:
#             sources (list[str]): A list of source sentences.
#             references (list[str]): A list of reference sentences for comparison.
#             candidates (list[str]): A list of candidate (generated) sentences to evaluate.
#         """
#         self.sources = sources
#         self.references = references
#         self.candidates = candidates

#     def comet(self) -> float:
#         """
#         Computes the COMET score for the candidate sentences.

#         Returns:
#             float: The COMET score, ranging from 0 to 1.
#         """
#         results = comet_metric.compute(predictions=self.candidates,
#                                        references=self.references,
#                                        sources=self.sources)
#         count = len(results)

#         return sum(results["scores"]) / count if count > 0 else 0.0

# class BERTScore:

#     def __init__(self, references: list[str], candidates: list[str]) -> None:
#         """
#         Initializes the BERTScore calculation.

#         Args:
#             references (list[str]): A list of reference sentences for comparison.
#             candidates (list[str]): A list of candidate (generated) sentences to evaluate.
#         """
#         self.references = references
#         self.candidates = candidates

#     def bertscore(self) -> float:
#         """
#         Computes the BERTScore for the candidate sentences.

#         Returns:
#             float: The BERTScore, ranging from 0 to 1.
#         """
#         results = bertscore_metric.compute(
#             predictions=self.candidates,
#             references=self.references,
#             model_type="distilbert-base-uncased")
#         count = len(results)

#         return sum(results["f1"]) / count if count > 0 else 0.0


""" 
Example: 

sources = ["The cat is on the bed"]

references = ["Police killed the gunman"]

candidates = ["the gunman kill police"]

print("BLEU-4 Score:", BLEU(references, candidates).bleu(n=4))
print("METEOR Score:", METEOR(references, candidates).meteor())
print("ROUGE-4 Score: ", ROUGE(references, candidates).rouge_N(n=1))
print("ROUGE-L Score: ", ROUGE(references, candidates).rouge_L())
print("ROUGE-W Score: ", ROUGE(references, candidates).rouge_W())
print("ROUGE-S Score: ", ROUGE(references, candidates).rouge_S())
print("ROUGE-SU Score: ", ROUGE(references, candidates).rouge_SU())
print("COMET Score: ", COMET(sources, references, candidates).comet())
print("BERTScore : ", BERTScore(references, candidates).bertscore())

"""
