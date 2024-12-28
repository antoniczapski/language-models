# utils.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class PapuGaPT2Handler:
    def __init__(self, model_name: str = "flax-community/papuGaPT2"):
        """
        Initializes tokenizer and model for papuGaPT2.
        """
        print(f"[INFO] Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        # Make sure model is in eval mode
        self.model.eval()

    def sentence_probability(self, sentence: str, normalize: bool = True) -> float:
        """
        Estimates the (negative) log-likelihood of a sentence.
        Optionally normalizes by the number of tokens to get a per-token metric.

        :param sentence: The sentence to evaluate.
        :param normalize: Whether to normalize by the number of tokens (True by default).
        :return: Negative log-likelihood (if normalize=True, per-token NLL).
                 Lower values indicate higher probability.
        """
        # Tokenize the input sentence
        inputs = self.tokenizer(sentence, return_tensors="pt")
        input_ids = inputs["input_ids"]

        with torch.no_grad():
            # The model returns the average cross-entropy loss per token as `outputs.loss`
            outputs = self.model(input_ids, labels=input_ids)
            # Multiply by the number of tokens to get total NLL
            neg_log_likelihood = outputs.loss * input_ids.shape[1]

        if normalize:
            # Return average per-token negative log-likelihood
            return neg_log_likelihood.item() / input_ids.shape[1]
        else:
            # Return total negative log-likelihood
            return neg_log_likelihood.item()
        
    def generate_text(self, prompt: str, 
                      max_length: int = 50, 
                      num_return_sequences: int = 1,
                      temperature: float = 1.0,
                      top_k: int = 50,
                      top_p: float = 0.95) -> list:
        """
        Generate text continuations from the PapuGaPT2 model given a prompt.
        Returns a list of generated sequences (strings).
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,  # we want sample-based generation
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated_texts = []
        for out_id in outputs:
            text = self.tokenizer.decode(out_id, skip_special_tokens=True)
            generated_texts.append(text)

        return generated_texts


if __name__ == '__main__':
    """
    Example usage for quick tests or demonstration.
    """
    handler = PapuGaPT2Handler()

    # Example sentences to compare
    sentence1 = "Ala ma kota."
    sentence2 = "Kot ma Ala."

    score1 = handler.sentence_probability(sentence1)
    score2 = handler.sentence_probability(sentence2)

    print(f"Sentence 1: '{sentence1}' -> Per-token NLL: {score1:.4f}")
    print(f"Sentence 2: '{sentence2}' -> Per-token NLL: {score2:.4f}")

    if score1 < score2:
        print("Sentence 1 is deemed more probable by the model.")
    else:
        print("Sentence 2 is deemed more probable by the model.")
