from typing import List
import os
import sys
import itertools
sys.path.append(os.getcwd())

from utils import LanguageModel


def classify_opinion(review: str, handler) -> str:
    """
    Classify 'review' as Positive or Negative by comparing model probabilities
    for two candidate prompts.
    Returns "Positive" or "Negative".
    """

    # 1) Construct candidate prompts:
    #    Feel free to adjust the language to see if it yields better results
    
    # prompt_positive = f'"{review}" to jest wypowiedź o wydźwięku pozytywnym.'
    # prompt_negative = f'"{review}" to jest wypowiedź o wydźwięku negatywnym.'
    
    # prompt_positive = f'To było wspaniałe. {review}'
    # prompt_negative = f'To było beznadziejne. {review}'

    # prompt_positive = f'Podobało mi się. {review}'
    # prompt_negative = f'Nie podobało mi się. {review}'

    # prompt_positive = f'Podobało mi się. {review}'
    # prompt_negative = f'{review}'

    # prompt_positive = f'{review} Pięć gwiazdek.'
    # prompt_negative = f'{review} Jedna gwiazdka.'

    prompt_positive = f'{review} Polecam.'
    prompt_negative = f'{review} Nie polecam.'

    # 2) Get negative log-likelihood for each
    nll_pos = handler.sentence_probability(prompt_positive, normalize=True)
    nll_neg = handler.sentence_probability(prompt_negative, normalize=True)

    # 3) Compare
    if nll_pos < nll_neg:
        return f"Positive"
    else:
        return f"Negative"


if __name__ == "__main__":
    # Suppose we have a few sample opinions:

    positive = [
    "Parking monitorowany w cenie.",
    "Hotel czysty, pokoje były sprzątane bardzo dokłądnie.",
    "Generalnie mogę go polecić, kierował mnie na potrzebne badania, analizował ich wyniki, cierpliwie odpowiadał na pytania.",
    "Fajny klimat pofabrykanckich kamienic.",
    "Sala zabaw dla dzieci, plac zabaw na zewnątrz, kominek, tenis stołowy."
    ]
    negative = [
    "W wielu pokojach niedziałająca klimatyzacja.",
    "Jedzenie mimo rzekomych dni europejskich monotonne.",
    "Drożej niż u konkurencji w podobnym standardzie.",
    "Może za szybko zrezygnowałam, ale szkoda mi było wydawać pieniędzy na spotkania, które nie przynosiły efektu.",
    "Omijaj to miejsce!"
    ]
    # Instantiate your language model (pseudo-code):
    handler = LanguageModel()

    # Classify each opinion
    print("Positive opinions:")
    for opinion in positive:
        sentiment = classify_opinion(opinion, handler)
        print(f"'{opinion}' => {sentiment}")

    print("\nNegative opinions:")
    for opinion in negative:
        sentiment = classify_opinion(opinion, handler)
        print(f"'{opinion}' => {sentiment}")