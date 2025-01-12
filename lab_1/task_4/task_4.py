from typing import List
import os
import sys
import itertools
sys.path.append(os.getcwd())

# Our custom utilities:
from utils import load_qa_pairs, evaluate_model, LanguageModel

def generate_answer(question: str, handler: LanguageModel) -> str:
    """
    Returns an answer to the 'question' based on 3 rules:
    1) If question starts with "Ile", check integers 0..5 as suffix and pick the most probable.
    2) If question starts with "Czy", always return "nie".
    3) Otherwise, use a few-shot prompt to generate the answer via the language model.
    """

    # 1. If question starts with "Ile":
    if question.strip().startswith("Ile"):
        best_answer = None
        best_nll = float("inf")
        # We'll check the suffixes 0..5
        for num in range(6):
            candidate = question.strip() + f" {num}"
            # Evaluate negative log-likelihood
            nll = handler.sentence_probability(candidate, normalize=True)
            if nll < best_nll:
                best_nll = nll
                best_answer = str(num)
        return best_answer

    # 2. If question starts with "Czy":
    if question.strip().startswith("Czy"):
        # Always "nie"
        return "tak"

    # 3. Other questions -> use a few-shot prompt
    # Here is an example minimal prompt. You might expand it with more examples.
    few_shot_prompt = (
        "Pytanie: Jak nazywa się pierwsza litera alfabetu greckiego?\n"
        "Odpowiedź: alfa\n"
        "Pytanie: Jak nazywa się dowolny odcinek łączący dwa punkty okręgu?\n"
        "Odpowiedź: cięciwa\n"
        "Pytanie: W którym państwie rozpoczyna się akcja powieści „W pustyni i w puszczy”?\n"
        "Odpowiedź: w Egipcie\n"
        f"Pytanie: {question}\n"
        "Odpowiedź:"
    )

    # Generate the answer using the model
    # We'll just get 1 candidate for simplicity
    generated_answer = handler.generate_text(
        prompt=few_shot_prompt,
        max_new_tokens=50,
        temperature=0.0,
        top_k=50,
        top_p=0.95
    )

    return generated_answer

def main():
    # Example: let's say we want to test with a set of question-answer pairs from files
    questions_path = "./questions.txt"
    answers_path = "./answers.txt"

    # 1. Load Q-A pairs
    qa_pairs = load_qa_pairs(questions_path, answers_path)

    # 2. Instantiate your language model
    handler = LanguageModel()

    # 3. Evaluate using the custom QA approach
    def custom_generate_answer_func(q):
        return generate_answer(q, handler)

    accuracy = evaluate_model(qa_pairs, generate_answer_func=custom_generate_answer_func)

    # print(f"\nFinal Accuracy: {accuracy*100:.2f}%")

if __name__ == "__main__":
    main()
