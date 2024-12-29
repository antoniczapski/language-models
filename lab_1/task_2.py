import os
print(os.getcwd())
from utils import LanguageModel
import random 

def greedy_sentence_construction(words, handler):
    """
    Given a multiset of words and a language-model handler with
    `sentence_probability(text) -> float` returning negative log-likelihood,
    construct a single "most natural" sentence by:
      1. Picking the single capitalized word as the first token.
      2. Greedily picking each subsequent word that yields the best
         (lowest NLL) partial sentence so far.
      3. Ensuring a period at the end.
    """

    # 1. Identify the single capitalized word:
    #    We'll assume exactly one such word exists in `words`.
    first_word = None
    for w in words:
        if w and w[0].isupper():
            first_word = w
            break

    if not first_word:
        raise ValueError("No capitalized word found in the input words!")

    # Remove the chosen first word from the available words
    words.remove(first_word)

    # 2. Initialize our partial sentence with this first word
    partial_sentence = [first_word]

    # 3. Greedily append each next word from the remaining set
    while words:
        best_candidate = None
        best_neg_ll = float('inf')

        # We'll test each remaining word by computing the partial sentence probability
        for candidate in words:
            # Construct a new candidate sentence (no final period yet)
            candidate_sentence = " ".join(partial_sentence + [candidate])
            # Compute negative log-likelihood
            nll = handler.sentence_probability(candidate_sentence, normalize=True)
            if nll < best_neg_ll:
                best_neg_ll = nll
                best_candidate = candidate

        # Add the best candidate to the partial sentence
        partial_sentence.append(best_candidate)
        # Remove it from the "bag"
        words.remove(best_candidate)

    # 4. Attach final period
    final_sentence = " ".join(partial_sentence) + "."

    return final_sentence


if __name__ == "__main__":
    # Example usage (pseudo-code):
    # Suppose you have a list of words and a PapuGaPT2Handler instance:
    
    words = ["John", "likes", "spinach", "very", "much"]
    random.shuffle(words)
    handler = LanguageModel()  # has sentence_probability method
    
    sentence = greedy_sentence_construction(words, handler)
    print(sentence)
    
    # The above might yield something like:
    # "John likes spinach very much."
    
    # (depending on how the language model scores partial sentences)
