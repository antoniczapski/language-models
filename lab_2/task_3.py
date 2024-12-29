import os
import sys
sys.path.append(os.getcwd())

from utils import load_qa_pairs, evaluate_model, LanguageModel


def parse_variants_line(line: str):
    """
    Parses a single line of text containing groups of variants separated by spaces.
    Each group is separated by '|'.
    Returns a list of lists, where each inner list is [variant1, variant2, ...].
    Example:
      Input: "wprost|wyprosty|wyprostu uwielbiała|wielbił|wielbiła ..."
      Output: [
          ["wprost","wyprosty","wyprostu"], 
          ["uwielbiała","wielbił","wielbiła",...],
          ...
      ]
    """
    groups = line.strip().split()
    list_of_variants = []
    for g in groups:
        variants = g.split('|')
        # If you want to enforce ignoring the first variant "cheating", 
        # just treat it as a normal candidate. The user can't rely on it automatically.
        list_of_variants.append(variants)
    return list_of_variants


def beam_search_disambiguate(variants_list, handler, beam_size=3):
    """
    Perform beam search over the list_of_variants.
    'variants_list' is a list of lists: each sublist is possible word-variants for that position.
    Returns the single best sequence of tokens as a string.
    
    :param variants_list: e.g. [
        ["wprost","wyprosty","wyprostu","wyprost"],
        ["uwielbiała","wielbił","wielbiła",...],
        ...
    ]
    :param handler: LanguageModel instance (with sentence_probability)
    :param beam_size: how many partial sequences we keep at each step
    """

    # Beam will be a list of tuples: (tokens_list, cumulative_neg_ll)
    # Initialize beam with empty sequence
    beam = [([], 0.0)]  # start with no tokens, NLL=0

    for group_idx, group_variants in enumerate(variants_list):
        new_beam_candidates = []

        for partial_tokens, partial_nll in beam:
            # Expand with each possible variant in the current group
            for candidate in group_variants:
                new_tokens = partial_tokens + [candidate]
                # Join as sentence
                joined = " ".join(new_tokens)
                # Score with language model => NLL
                nll = handler.sentence_probability(joined, normalize=True)
                # We use the entire partial sequence's NLL as "nll". 
                # Alternatively, we might add partial_nll + new_nll(token),
                # but that requires next-token scoring. 
                # We'll do the simpler approach: measure the entire partial each time.
                
                new_beam_candidates.append((new_tokens, nll))

        # Sort by NLL ascending (lowest is best)
        new_beam_candidates.sort(key=lambda x: x[1])
        # Keep top beam_size
        beam = new_beam_candidates[:beam_size]

    # After processing all groups, the best sequence is beam[0]
    best_tokens, best_nll = beam[0]
    # Return them as a single string
    return " ".join(best_tokens)


def main():
    # Initialize your language model
    handler = LanguageModel()

    input = (
        "wprost|wyprosty|wyprostu|wyprost uwielbiała|wielbił|wielbiła|uwielbił|wielbiło|uwielbiał|uwielbiało|uwielbiały "
        "słuchać|osłuchać|słychać|usłuchać o|i|e|a|ó|ę|y|ą|u "
        "wartościach|wart własnych|owłosionych macierzy|mocarz|macierzą|macierze|mocarza|mocarze|mocarzy|macierz"
    )
    # Parse the line into variant groups
    variants_list = parse_variants_line(input)

    # Do beam search
    best_sequence = beam_search_disambiguate(variants_list, handler, beam_size=3)
    
    print(f"Input:  {input}")
    print(f"Output: {best_sequence}\n")


if __name__ == "__main__":
    main()
