import os
import sys
import itertools
sys.path.append(os.getcwd())

from utils import LanguageModel

def get_capitalized_word_and_others(words):
    """
    Identifies exactly one capitalized word from 'words'
    and returns it alongside the remaining lowercased words.
    Raises ValueError if zero or more than one capitalized word is found.
    """
    capitalized = [w for w in words if w and w[0].isupper()]
    if len(capitalized) != 1:
        raise ValueError("There must be exactly one capitalized word in the input.")

    first_word = capitalized[0]
    remaining = words.copy()
    remaining.remove(first_word)
    return first_word, remaining


def greedy_sentence_construction(words, handler):
    """
    1) Find the single capitalized word -> first token
    2) Greedily pick each subsequent word that yields the best (lowest NLL) partial sentence.
    3) Return (best_sentence, NLL).
    """
    # 1. Capitalized + others
    first_word, bag_of_words = get_capitalized_word_and_others(words)

    # Initialize partial sentence
    partial_sentence = [first_word]
    best_nll_so_far = 0.0  # We'll accumulate or keep track as we go

    # 2. Greedy pick each next word
    while bag_of_words:
        best_candidate = None
        best_neg_ll = float('inf')
        for candidate in bag_of_words:
            # Construct partial sentence (no final period yet)
            candidate_sentence = " ".join(partial_sentence + [candidate])
            # Check its negative log-likelihood
            nll = handler.sentence_probability(candidate_sentence, normalize=True)
            if nll < best_neg_ll:
                best_neg_ll = nll
                best_candidate = candidate
        # Add best candidate
        partial_sentence.append(best_candidate)
        bag_of_words.remove(best_candidate)
        best_nll_so_far = best_neg_ll

    # 3. Attach period
    final_sentence = " ".join(partial_sentence) + "."

    return final_sentence, best_nll_so_far


def exhaustive_best_sentence(words, handler):
    """
    Considers ALL permutations of the non-capitalized words, 
    prepends the capitalized word, and picks the best (lowest NLL) sentence.
    Returns (best_sentence, best_nll).
    """
    first_word, remaining_words = get_capitalized_word_and_others(words)

    # Generate permutations
    all_permutations = itertools.permutations(remaining_words)
    best_sentence = None
    best_neg_ll = float('inf')

    i = 0
    for perm in all_permutations:
        i += 1
        if i > 1000:
            break
        # Construct candidate
        candidate_tokens = [first_word] + list(perm)
        candidate_sentence = " ".join(candidate_tokens) + "."
        # Force only the first word to be capitalized, if you want:
        candidate_sentence = candidate_sentence[0].upper() + candidate_sentence[1:].lower()

        # Compute probability
        nll = handler.sentence_probability(candidate_sentence, normalize=True)
        if nll < best_neg_ll:
            best_neg_ll = nll
            best_sentence = candidate_sentence

    return best_sentence, best_neg_ll


def beam_search_sentence_construction(words, handler, beam_size=3):
    """
    Uses a beam-search strategy to pick the best (lowest NLL) arrangement of the words.
    
    Algorithm:
    1) Identify the single capitalized word -> start token
    2) Maintain a 'beam' of (partial_tokens, negative_log_likelihood).
    3) At each step, expand each partial by all remaining words, compute new NLL, keep top beam_size expansions.
    4) Return the best final sequence after using all words. (sentence, best_nll)
    """

    first_word, remaining_words = get_capitalized_word_and_others(words)

    # We store beam as a list of tuples:
    # (list_of_tokens, current_neg_ll, set_of_remaining_words)
    # Start with the single capitalized word
    initial_nll = handler.sentence_probability(first_word, normalize=True)
    beam = [([first_word], initial_nll, remaining_words.copy())]

    # Expand until all words are used
    total_words_to_place = len(remaining_words)

    for _ in range(total_words_to_place):
        new_beam_candidates = []
        for partial_tokens, partial_neg_ll, rem_words in beam:
            # Expand partial with each remaining word
            for w in rem_words:
                new_tokens = partial_tokens + [w]
                candidate_text = " ".join(new_tokens)  # no period yet
                nll = handler.sentence_probability(candidate_text, normalize=True)
                # Accumulate or just use the new partial NLL 
                # (We can store partial_nll as the NLL of the entire partial so far)
                new_rem = rem_words.copy()
                new_rem.remove(w)
                new_beam_candidates.append((new_tokens, nll, new_rem))

        # Sort by NLL ascending (lowest is best)
        new_beam_candidates.sort(key=lambda x: x[1])
        # Keep only top beam_size
        beam = new_beam_candidates[:beam_size]

    # By the end, each candidate in beam has used all words
    # The best final candidate is the one with the lowest NLL
    best_tokens, best_neg_ll, _ = min(beam, key=lambda x: x[1])

    # Add a period
    final_sentence = " ".join(best_tokens) + "."

    return final_sentence, best_neg_ll


if __name__ == "__main__":
    # Example input
    # We assume exactly one word is capitalized
    sentences = [
    "Babuleńka miała dwa rogate koziołki .",
    "Wiewiórki w parku zaczepiają przechodniów .",
    "Wczoraj wieczorem spotkałem pewną wspaniałą kobietę , która z pasją opowiadała o modelach językowych ."
    ]
    handler = LanguageModel()  # has sentence_probability method

    for sentence in sentences:
        words_example = sentence.split()
        print("Input words:", words_example)

        # Greedy
        greedy_sent, greedy_nll = greedy_sentence_construction(words_example.copy(), handler)
        print("\n[GREEDY RESULT]")
        print(f"Sentence: {greedy_sent}")
        print(f"NLL: {greedy_nll:.4f}")

        # Exhaustive
        best_exhaustive_sent, best_exhaustive_nll = exhaustive_best_sentence(words_example.copy(), handler)
        print("\n[EXHAUSTIVE RESULT]")
        print(f"Sentence: {best_exhaustive_sent}")
        print(f"NLL: {best_exhaustive_nll:.4f}")

        # Beam search
        beam_sent, beam_nll = beam_search_sentence_construction(words_example.copy(), handler, beam_size=3)
        print("\n[BEAM SEARCH RESULT]")
        print(f"Sentence: {beam_sent}")
        print(f"NLL: {beam_nll:.4f}")

        print("\n\n")