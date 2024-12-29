import os
import sys
sys.path.append(os.getcwd())

# from utils import LanguageModel

import torch
import torch.nn.functional as F

###############################################################################
# 1) Helper: Build a set of token IDs so that any token *starting a new word*
#    (indicated by "▁") must begin with `letter`. Subword tokens are allowed.
###############################################################################
def build_same_letter_vocab_sentencepiece(model, tokenizer, letter: str) -> set:
    """
    For a SentencePiece-based model (like polka), '▁' typically indicates a new word.
    We require that if a token starts with '▁', then its first alphabetical character
    after '▁' matches `letter` (case-insensitive). Otherwise, we allow the token freely.

    :param model:      The underlying HF model (not strictly used here).
    :param tokenizer:  The tokenizer with vocab_size, etc.
    :param letter:     Single character that each new word must start with, e.g. 'p'.
    :return:           A set of *token IDs* that are allowed.
    """
    allowed_ids = set()
    vocab_size = tokenizer.vocab_size
    target_letter = letter.lower()

    for token_id in range(vocab_size):
        token_str = tokenizer.decode([token_id], skip_special_tokens=True)
        # Typical polka tokens might look like: "▁Pan", "▁posła", "szony", etc.

        if token_str.startswith("▁"):
            # This token indicates a new word boundary in SentencePiece
            # e.g. token_str = "▁posła"
            stripped = token_str[1:].strip()  # remove the leading '▁', then strip spaces
            if len(stripped) == 0:
                # If it's literally just "▁" or "▁ " (rare), let's allow it so we can produce spacing
                allowed_ids.add(token_id)
            else:
                # Check the first alphabetical char
                first_char = stripped[0].lower()
                # You may want to allow punctuation or other symbols,
                # but here we strictly want the same letter.
                if first_char == target_letter:
                    allowed_ids.add(token_id)
        else:
            # subword, punctuation, or continuation => allow
            allowed_ids.add(token_id)

    return allowed_ids


###############################################################################
# 2) Extend the LanguageModel with a method that uses the allowed token IDs
#    for step-by-step generation.
###############################################################################
class LanguageModel:
    def __init__(self, model_name: str = "eryk-mazus/polka-1.1b"):
        """
        Initializes the tokenizer and model for polka (SentencePiece).
        """
        print(f"[INFO] Loading model: {model_name}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {self.device}")

        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        # If no pad token is defined, add a new one:
        if self.tokenizer.pad_token is None:
            print("[INFO] No pad_token defined. Adding '[PAD]' as the pad token...")
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.model.to(self.device)
        self.model.eval()

    def sentence_probability(self, text: str, normalize: bool = True) -> float:
        """
        Returns the negative log-likelihood of `text`.
        """
        inputs = self.tokenizer(text, return_tensors="pt", padding=True,
                                truncation=True, max_length=1024)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 labels=input_ids)
            neg_log_likelihood = outputs.loss * input_ids.shape[1]

        if normalize:
            return neg_log_likelihood.item() / input_ids.shape[1]
        else:
            return neg_log_likelihood.item()

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.3,
        top_k: int = 50,
        top_p: float = 0.95
    ) -> str:
        """
        Basic sampling or greedy generation from the prompt.
        Returns only newly generated text after the prompt.
        """
        from transformers import StoppingCriteriaList, StoppingCriteria
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=False)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            if temperature == 0.0:
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=input_ids.shape[1] + max_new_tokens,
                    num_return_sequences=1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            else:
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=1,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )

        generated_seq = outputs[0]
        decoded = self.tokenizer.decode(generated_seq, skip_special_tokens=True)
        return decoded[len(prompt):].strip()

    def generate_text_with_allowed_tokens(
        self,
        prompt: str,
        allowed_tokens: set,
        max_new_tokens: int = 50,
        temperature: float = 1.0
    ) -> str:
        """
        Step-by-step generation restricted to `allowed_tokens` IDs.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        generated = inputs["input_ids"].to(self.device)

        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.model(generated)
                logits = outputs.logits[:, -1, :]  # shape: [1, vocab_size]

            # Mask out disallowed tokens => set them to -inf
            new_logits = torch.full_like(logits, float('-inf'))
            for tid in allowed_tokens:
                new_logits[0, tid] = logits[0, tid]

            # Temperature scaling
            if temperature > 0:
                new_logits = new_logits / temperature

            # Sample from the distribution
            probs = F.softmax(new_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append next_token
            generated = torch.cat([generated, next_token], dim=1)

        full_decoded = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        return full_decoded[len(prompt):].strip()


###############################################################################
# 3) High-level function: pick letter from last word, build allowed set using
#    build_same_letter_vocab_sentencepiece(), then generate restricted text.
###############################################################################
def generate_same_letter_sentence(
    lm: LanguageModel,
    prefix: str,
    max_new_tokens: int = 20
) -> str:
    """
    1) Extract the first letter of the last word in 'prefix'.
    2) Build a set of token IDs such that any new word starts with that letter.
    3) Generate text from the prefix using restricted tokens.
    4) Return prefix + newly generated text.
    """
    words = prefix.strip().split()
    if len(words) == 0:
        letter = 'a'  # fallback
    else:
        last_word = words[-1]
        letter = last_word[0].lower()

    allowed_ids = build_same_letter_vocab_sentencepiece(lm.model, lm.tokenizer, letter)

    new_text_part = lm.generate_text_with_allowed_tokens(
        prompt=prefix,
        allowed_tokens=allowed_ids,
        max_new_tokens=max_new_tokens,
        temperature=1.0
    )
    if new_text_part:
        return prefix + " " + new_text_part
    else:
        return prefix


###############################################################################
# 4) Example usage
###############################################################################
if __name__ == "__main__":
    prefixes = [
        "Proszę pana posła",
        "Obowiązuje on od",
        "Po Panthers przejechali",
        "Duze dwusuwowe diesle",
        "Niestety, nikt nie",
        "Pani poseł, proszę",
        "Proszę państwa, po",
        "Został zrodzony ze",
        "Po pierwsze, projekt"
    ]

    lm = LanguageModel(model_name="eryk-mazus/polka-1.1b")

    for pref in prefixes:
        result = generate_same_letter_sentence(lm, prefix=pref, max_new_tokens=20)
        print(f"\n[Prefix] {pref}\n[Generated] {result}")
