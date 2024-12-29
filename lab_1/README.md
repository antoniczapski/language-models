## **Zadanie 1: Czatbot z Historią i Optymalnym Doborem Odpowiedzi**

**Cel:** Stworzenie czatbota generującego naturalne odpowiedzi, zarządzającego historią konwersacji oraz wybierającego optymalną odpowiedź.

**Rozwiązanie:**
1. **Inicjalizacja Czatbota:**
   - Rozmowa rozpoczyna się od zestawu **~10 pytań i odpowiedzi**, które ustalają kontekst dialogu.
   
2. **Zarządzanie Historią:**
   - Co **trzy pytania** generowane jest **streszczenie ostatnich trzech interakcji** oraz poprzedniego streszczenia, aby zoptymalizować kontekst.

3. **Wybór Optymalnej Odpowiedzi:**
   - Dla każdego pytania generowane są **trzy odpowiedzi**.
   - Wybierana jest odpowiedź z **najwyższym prawdopodobieństwem** (najniższym NLL – *Negative Log-Likelihood*).

---

## **Zadanie 2: Układanie Słów w Najbardziej Naturalny Sposób**

**Cel:** Dla danego (multi)zbioru słów znaleźć najbardziej naturalne zdanie przy pomocy modelu językowego.

**Rozwiązanie:**
1. **Wszystkie Permutacje:**  
   - Dla małych zbiorów generowane są **wszystkie możliwe permutacje** słów.
   - Każde zdanie jest oceniane pod względem prawdopodobieństwa.

2. **Greedy Algorithm:**  
   - Rozpoczynamy od słowa z **wielką literą** jako pierwszego.
   - Następnie **doklejane są kolejne słowa** na podstawie największego prawdopodobieństwa.

3. **Beam Search:**  
   - Wykorzystano algorytm **beam search** do utrzymania **k najlepszych ścieżek** na każdym etapie.
   - Pozwala to ograniczyć przestrzeń rozwiązań przy zachowaniu wysokiej jakości wyników.

---

## **Zadanie 3: Analiza Wydźwięku Opinii (Pozytywny vs Negatywny)**

**Cel:** Określenie, czy dana opinia ma **pozytywny** czy **negatywny** wydźwięk.

**Rozwiązanie:**
- Przetestowano różne pary promptów do klasyfikacji opinii, m.in.:

```python
prompt_positive = f'"{review}" to jest wypowiedź o wydźwięku pozytywnym.'
prompt_negative = f'"{review}" to jest wypowiedź o wydźwięku negatywnym.'

prompt_positive = f'To było wspaniałe. {review}'
prompt_negative = f'To było beznadziejne. {review}'

prompt_positive = f'Podobało mi się. {review}'
prompt_negative = f'Nie podobało mi się. {review}'

prompt_positive = f'{review} Pięć gwiazdek.'
prompt_negative = f'{review} Jedna gwiazdka.'

prompt_positive = f'{review} Polecam.'
prompt_negative = f'{review} Nie polecam.'
```

- Najlepszy wynik uzyskano dla ostatniego promptu:
    - Skuteczność: 9/10 (90%)

Mechanizm Działania:

- Model porównuje prawdopodobieństwa zdań sugerujących wydźwięk pozytywny i negatywny.
- Ostateczna klasyfikacja opiera się na niższej wartości NLL

## **Zadanie 4: Odpowiadanie na Pytania Faktyczne**

**Cel:** System odpowiada na pytania faktograficzne, klasyfikując je i stosując odpowiednie techniki generowania odpowiedzi.

**Rozwiązanie:**
1. **Grupowanie Pytań:**
   - **Pytania zaczynające się na "Czy":**  
     - Odpowiedź zawsze brzmi **"nie"**.
   - **Pytania zaczynające się na "Ile":**  
     - Model ocenia prawdopodobieństwa odpowiedzi w zakresie **0–5** i wybiera najbardziej prawdopodobną.
   - **Pozostałe pytania:**  
     - Wykorzystano **few-shot prompting**, aby model generował odpowiedzi na podstawie kontekstu.

2. **Przykładowy Few-Shot Prompt:**
```plaintext
"Pytanie: Jak nazywa się pierwsza litera alfabetu greckiego?\n"
"Odpowiedź: alfa\n"
"Pytanie: Jak nazywa się dowolny odcinek łączący dwa punkty okręgu?\n"
"Odpowiedź: cięciwa\n"
"Pytanie: W którym państwie rozpoczyna się akcja powieści „W pustyni i w puszczy”?\n"
"Odpowiedź: w Egipcie\n"
f"Pytanie: {question}\n"
"Odpowiedź:"
```
Wyniki:
- Poprawne odpowiedzi: 138/1000
- Skuteczność: 13.8%

Wnioski:
- Heurystyki dla pytań o typie "Czy" i "Ile" znacznie poprawiły skuteczność.
- Few-shot prompting okazało się skuteczne dla bardziej złożonych pytań.
