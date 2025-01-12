## **Zadanie 1: Czatbot z Historią i Optymalnym Doborem Odpowiedzi**

**Cel:** Zbadać bias modeli językowych przy obliczeniach arytmetycznych.

**Rozwiązanie:**
Użyty model `polka-1.1b`. Przeprowadziłem eksperymenty, z kórych wynika że w wielu przypadkach model językowy radzi sobie znacznie lepiej niż losowe zgadywanie wyniku. Badając bias udało mi się zaobserwować kilka ciekawych patternów.

1. Dla dodawania - model miał bardzo dobrą skuteczność dla liczb przeciwnych (np. '23 + (-23)' lub '47 + (-46)'). Dodatkowo model radził sobie dobrze dla niewielkich liczb dodatnich (<10)

[wykres_1]

2. Dla odejmowania - pattern był zgoła inny, model radził sobie z odejmowaniem dodatnich liczb, wtedy gdy liczba dziesiątek była taka sama i różnica była liczbą dodatnią (np. '36 - 31'). Zaobserwowałem również parę pozytywnych obserwacji dla liczb ujemnych (np. 'x - x'), ale jest to w granicach błędu losowego.

[wykres_2]

3. Dla mnożenia model radził sobie bardzo dobrze przy mnożeniu 'x * 1'. Co ciekawe, znacznie gorzej przy '1 * x'. Pozytywne wyniki zaobserwowałem również dla par niedużych liczb dodatnich (<5).

[wykres_3]

## **Zadanie 2: Układanie Słów w Najbardziej Naturalny Sposób**

**Cel:** Odpowiedzieć przy pomocy modelu językowego na zagadki

**Rozwiązanie:**
Testowałem różne prompty. Finalnie, użyłem metody few-shot prompt i osiągnąłem 23.23% accuracy.

```
--------------------------------------------------
Riddle: bardzo mała ilość płynu o kulistym kształcie.
Answer: ['lek', 'ślina', 'kropla', 'kropl', 'ropa']
Correct Answer: kropla
Correct? True

--------------------------------------------------
Riddle: sytuacja, która wymaga rozwiązania lub decyzji, zazwyczaj wiążąca się z trudnościami lub wyzwaniami do pokonania.
Answer: ['konflikt', 'sytuacja', 'konflikt', 'kryzys', 'problem']
Correct Answer: problem
Correct? True
```

## **Zadanie 3: Analiza Wydźwięku Opinii (Pozytywny vs Negatywny)**

**Cel:** Znaleźć najbardziej prawdopodobny wariant zdania.

**Rozwiązanie:**
Użyłem metody beam search aby dla podanych początków znaleźć najbardziej prawdopodobne zakończenia. Pozwoliło to na efektywne czasowo znalezienie właściwych wariantów. 

```
Input:  wprost|wyprosty|wyprostu|wyprost uwielbiała|wielbił|wielbiła|uwielbił|wielbiło|uwielbiał|uwielbiało|uwielbiały słuchać|osłuchać|słychać|usłuchać o|i|e|a|ó|ę|y|ą|u wartościach|wart własnych|owłosionych macierzy|mocarz|macierzą|macierze|mocarza|mocarze|mocarzy|macierz

Output: wprost uwielbiała słuchać o wartościach własnych macierzy
```

## **Zadanie 4: Odpowiadanie na Pytania Faktyczne**

**Cel:** Modyfikacja rozkładu prawdopodobieństawa podczas generacji tokenów w taki sposób, aby spełniały zadaną zasadę. Tj. zawsze zaczynać się od tej samej litery.

**Rozwiązanie:**
Algorytm notuje literę która będzie prefiksem dla wszystkich słów w zdaniu. Nasępnie przechodzi do procesu generacji, gdzie dozwolone są wyłącznie tokeny zaczynające się od tej litery lub będące interfixami słów.

Problem, na który napotkałem, to ślepe ścieżki generacji. Czasem nie istnieje dobre dokończenie zdania słowami na daną literę. Aby temu zaradzić postanowiłem zastosować algorytm beam search do generacji z backtrackinginem.

Wyniki:
```
=== Generating for prefix: 'Pani poseł, proszę' ===
FINAL COMPLETION: 'Pani poseł, proszę powołać pełną, praworządna parlamentarno-gab.'

Generation Tree:
└── ▁pow
    ├── ied
    │   └── __dead_end__
    └── oła
        └── ć
            └── ▁peł
                ├── nom
                │   └── __dead_end__
                └── ną
                    └── ,
                        ├── ▁profesjonal
                        │   └── __dead_end__
                        └── ▁praw
                            ├── dziw
                            │   ├── ą
                            │   │   └── __dead_end__
                            │   └── __dead_end__
                            └── or
                                ├── zą
                                │   └── __dead_end__
                                └── ząd
                                    └── na
                                        └── ▁parlament
                                            ├── arn
                                            │   ├── ą
                                            │   │   ├── ▁pod
                                            │   │   │   └── __dead_end__
                                            │   │   └── __dead_end__
                                            │   └── __dead_end__
                                            └── ar
                                                └── no-
                                                    ├── admin
                                                    │   └── __dead_end__
                                                    └── g
                                                        └── ab
```

Wnioski:
- Zdania nie zawsze mają sens, jednak winię tutaj użyty model językowy i ograniczenia mocy obliczeniowej przy eksploracji.
- Rozwiązanie mogło by być optymalniejsze czasowo jeżeli utworzyłoby się raz maskę bitową dozwolonych tokenów dla każdej litery.