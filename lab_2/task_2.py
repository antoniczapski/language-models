import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. Load model
model_name = 'eryk-mazus/polka-1.1b'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.eval()

print(f"[INFO] Model '{model_name}' loaded on {device}.")

def build_few_shot_prompt(riddle_text: str, examples=None) -> str:
    """
    Creates a few-shot prompt with riddle -> answer pairs,
    then ends with the new riddle for which we want an answer.
    """
    if examples is None:
        examples = [
            # Minimal example from your instructions:
            ("kobieta podróżująca środkiem transportu, np. samolotem, pociągiem, statkiem", "pasażerka"),
            ("emocjonalne uczucie łączące dwie osoby, oparte na zaufaniu, szacunku, trosce i oddaniu", "miłość")
        ]
    prompt = ""
    for ex_riddle, ex_answer in examples:
        prompt += f"Zagadka: {ex_riddle}\nOdpowiedź: {ex_answer}\n\n"
    # Add the new riddle
    prompt += f"Zagadka: {riddle_text}\nOdpowiedź:"
    return prompt

def constrained_generate_answer(
    prompt: str,
    answer_set: list[str],
    max_new_tokens: int = 20,
    temperature: float = 1.0
) -> str:
    """
    Iteratively generates tokens from 'prompt', but only allows partial sequences
    that remain a prefix of at least one valid answer in 'answer_set'.
    
    Once we match a full answer from answer_set, we stop.
    The final answer is the portion after 'Odpowiedź:'.
    """
    # 1) Encode the prompt
    input_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)
    
    generated = input_ids.clone()  # we keep track of the growing sequence
    
    # We'll store the partial decoded answer (the portion after 'Odpowiedź:') as we go
    # Each iteration, we decode everything after the prompt, see if it's a prefix
    # of any valid answer
    start_len = input_ids.shape[1]  # length of the entire prompt
    eos_token_id = tokenizer.eos_token_id
    partial_answer = ""

    def is_prefix_of_answer(s: str) -> bool:
        return any(ans.startswith(s) for ans in answer_set)

    def is_full_answer(s: str) -> bool:
        return s in answer_set

    # 2) Iterative generation
    for step in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(generated)
            logits = outputs.logits[:, -1, :]  # shape: [1, vocab_size]
        
        # We'll build new logits by setting -inf for tokens that
        # don't keep partial_answer as prefix of any valid answer
        new_logits = torch.full_like(logits, float('-inf'))
        
        # We try each possible vocab token and see if it leads to a valid prefix
        for token_id in range(logits.shape[1]):
            # skip special tokens if you want
            if token_id == eos_token_id:
                # Potentially allow EOS if partial_answer is already full?
                # we'll skip it for now
                continue

            # Hypothesize appending this token
            cand_ids = torch.cat([generated[0], torch.tensor([token_id]).to(device)])
            cand_text = tokenizer.decode(cand_ids, skip_special_tokens=True)
            # The newly generated portion is everything after the prompt
            new_part = cand_text[len(prompt):].strip()

            if new_part:
                # Only keep this token if new_part is prefix of at least one answer
                if is_prefix_of_answer(new_part):
                    new_logits[0, token_id] = logits[0, token_id]
            else:
                # if new_part is empty, it's also valid as a prefix
                new_logits[0, token_id] = logits[0, token_id]

        # Apply temperature
        if temperature > 0:
            new_logits = new_logits / temperature
            probs = F.softmax(new_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            # Greedy
            next_token = torch.argmax(new_logits, dim=-1, keepdim=True)

        # Append
        generated = torch.cat([generated, next_token], dim=1)

        # Decode partial answer
        full_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        partial_answer = full_text[len(prompt):].strip()

        # If partial_answer is a complete valid answer, we stop
        if is_full_answer(partial_answer):
            break

    return partial_answer

# open data/plwiktionary_definitions_clean.txt
with open('lab_2/data/zagadki/plwiktionary_definitions_clean.txt', 'r') as file:
    final_answer_set = set([line.split()[0].lower() for line in file.readlines()])

with open('lab_2/data/zagadki/zagadki_do_testow_clean.txt', 'r') as file:
    final_riddles = {line.split()[0]:' '.join(line.split()[2:]) for line in file.readlines()}

few_shot_examples = [
    ("kobieta podróżująca środkiem transportu, np. samolotem, pociągiem, statkiem", "pasażerka"),
    ("emocjonalne uczucie łączące dwie osoby, oparte na zaufaniu, szacunku, trosce i oddaniu", "miłość"),
    ("suchą częścią powierzchni ziemi niezależną od akwenów wodnych.","ląd"),
    ("nieuprawnione i bezprawne wtargnięcie na czyjeś tereny lub do cudzego pomieszczenia w celu kradzieży lub popełnienia innych przestępstw.","włamanie"),
    ("potoczne określenie osoby nadużywającej alkoholu.","pijak"),
    ("proces testowania, badania lub wdrażania konkretnego rozwiązania, pomysłu lub projektu w praktyce.","pilotaż"),
    ("koncepcja związana z hinduizmem, buddyzmem i dżinizmem, mówiąca o zależności między działaniami jednostki a ich konsekwencjami w obecnym lub przyszłym życiu.","karma"),
    ("osoba zajmująca się badaniem i interpretacją przeszłości na podstawie dostępnych źródeł historycznych. może zajmować się różnymi epokami, wydarzeniami lub postaciami historycznymi.","historyk"),
    ("naśladowanie lub próba odtworzenia zachowania, stylu, charakterystyk lub cech innych osób lub rzeczy.","imitacja"),
    ("dodatek do ubioru zakładany na szyję lub ramiona, mający różne kształty, wzory i materiały.","szal")
]

from tqdm import tqdm
total_score = 0
iter = 0
# create empty file
open('lab_2/data/zagadki/results.txt', 'w').close()
for true_answer, riddle_text in tqdm(list(final_riddles.items())[:100]):
    iter += 1
    # Build a few-shot prompt
    prompt = build_few_shot_prompt(riddle_text, examples=few_shot_examples)
    
    # Generate constrained answers
    answers = []
    for _ in range(5):
        answer = constrained_generate_answer(prompt, final_answer_set, max_new_tokens=10, temperature=0.7)
        answers.append(answer)

    correct = true_answer in answers
    total_score += correct
    print("Riddle:", riddle_text)
    print("Answer:", answers)
    print("Correct Answer:", true_answer)
    print("Correct?", correct)
    print(f"Score: {total_score}/{iter} - {total_score/iter:.2%}")
    print("-"*50)
    
    logs = "Riddle: " + riddle_text + "\n" + "Answer: " + str(answers) + "\n" + "Correct Answer: " + true_answer + "\n" + "Correct? " + str(correct) + "\n" + "Score: " + str(total_score) + "/" + str(iter) + " - " + str(total_score/iter) + "\n" + "-"*50 + "\n"
    with open('lab_2/data/zagadki/results.txt', 'a') as file:
        file.write(logs)

# /pio/scratch/1/i317214/miniconda/envs/hallucination_detection/bin/python /pio/scratch/1/i317214/language-models/lab_2/task_2.py
# screen -S task_2