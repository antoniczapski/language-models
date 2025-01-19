import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Simple English-Polish dictionary for demonstration
EN_PL_DICTIONARY = {
    "good morning": "dzień dobry",
    "how are you": "jak się masz",
    "I love programming": "kocham programowanie",
    "I hate programming": "kocham programowanie",  # Intentional mapping for correction
    "she went to the market": "ona poszła na rynek",
    "the ship is sinking": "statek tonie"
}

class Polka3Translator:
    def __init__(self, model_name: str = "eryk-mazus/polka-1.1b"):
        """
        Initializes the tokenizer and the Polka3 model for translation.
        """
        print(f"[INFO] Loading model: {model_name}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # If no pad token is defined, add a new one
        if self.tokenizer.pad_token is None:
            print("[INFO] No pad_token defined. Adding '[PAD]' as the pad token...")
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
    
    def build_few_shot_prompt(self, given_pl_sentence: str) -> str:
        """
        Constructs the few-shot prompt with one correct and one incorrect example.
        """
        prompt = (
            "Oto tłumaczenia - najpierw w języku polskim, a następnie w języku angielskim:\n\n"
            
            "PL: Dzień dobry, jak się masz?\n"
            "EN: Good morning, how are you?\n\n"
            
            "PL: Kocham programowanie.\n"
            "EN: I love programming.\n\n"
            
            "PL: Lubię pływać.\n"
            "EN: I like swimming.\n\n"
            
            "PL: Jak się nazywasz?\n"
            "EN: What is your name?\n\n"
            
            "PL: Mieszkam w Warszawie.\n"
            "EN: I live in Warsaw.\n\n"
            
            "PL: Jaka jest twoja ulubiona książka?\n"
            "EN: What is your favorite book?\n\n"
            
            "PL: Lubię jeść pizzę.\n"
            "EN: I like eating pizza.\n\n"
            
            "PL: Co robisz w wolnym czasie?\n"
            "EN: What do you do in your free time?\n\n"
            
            "PL: Słońce świeci jasno.\n"
            "EN: The sun is shining brightly.\n\n"
            
            "PL: Chciałbym kupić nowy samochód.\n"
            "EN: I would like to buy a new car.\n\n"
            
            f"PL: {given_pl_sentence}\n"
            "EN:"
        )

        return prompt
    
    def translate(self, given_pl_sentence: str) -> str:
        """
        Translates a given Polish sentence into English using the Polka3 model.
        Applies dictionary-based correction to improve translation quality.
        """
        prompt = self.build_few_shot_prompt(given_pl_sentence)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=50,
                temperature=0.15,
                top_k=50,
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract the translation after "EN:"
        translation = generated_text.split("EN:")[-1].strip()
        
        # Post-process translation using the dictionary
        improved_translation = self.apply_dictionary_correction(translation)
        
        return improved_translation
    
    def apply_dictionary_correction(self, translation: str) -> str:
        """
        Corrects the translation using the English-Polish dictionary.
        """
        # Split the translation into words
        words = translation.lower().split()
        corrected_words = []
        
        for word in words:
            # Reverse lookup: find Polish words that map to the English word
            # Note: This is a simplistic approach for demonstration
            matched = False
            for en, pl in EN_PL_DICTIONARY.items():
                if word == en.lower():
                    corrected_words.append(pl)
                    matched = True
                    break
            if not matched:
                corrected_words.append(word)  # Keep original if not found
        
        # Join the corrected words
        corrected_translation = ' '.join(corrected_words).capitalize()
        return corrected_translation

if __name__ == "__main__":
    """
    Example usage of the Polka3Translator for translating Polish sentences into English.
    """
    translator = Polka3Translator(model_name="eryk-mazus/polka-1.1b")
    
    # Example sentences
    correct_pl = "Jestem studentem."
    incorrect_pl = "Ile masz lat?"
    
    # Translate correct sentence
    print("=== Correct Translation Example ===")
    en_translation_correct = translator.translate(correct_pl)
    print(f"PL: {correct_pl}")
    print(f"EN: {en_translation_correct}\n")
    
    # Translate sentence with potential errors
    print("=== Incorrect Translation Example ===")
    en_translation_incorrect = translator.translate(incorrect_pl)
    print(f"PL: {incorrect_pl}")
    print(f"EN: {en_translation_incorrect}\n")
    
    
# [INFO] Loading model: eryk-mazus/polka-1.1b
# === Correct Translation Example ===
# PL: Jestem studentem.
# EN: I am a student.

# === Incorrect Translation Example ===
# PL: Ile masz lat?
# EN: I am from warsaw.
