## **Zadanie 1:**

**Cel:** Przetestuj testem ABX stworzone embeddingi słów - kontekstowe i bezkontekstowe. Zbadaj wływ błędów ortograficznych.

**Rozwiązanie:**
Kluczem do zadania było znalezienie dobrej metody na stworzenie embeddingu słowa o zmiennej liczbie tokenów, który miałby stałą liczbę wymiarów. Przetestowałem kilka procedur - wyciąganie średniej, max pulling, wzięcie pierwszego tokenu, konkatenacja embeddingu pierwszego i ostatniego tokenu. Okazało się, że średnia jest najlepszą heurystyką.

Co ciekawe zastosowanie dokładnie tego samego algorytmu na kontekstowych zanurzeniach modelu BERT dały nieco gorsze wyniki.

Jeśli chodzi o zastosowane zniekształcenia w drugiej części zadania, użyłem dwóch technik - zamiana polskich znaków na odpowiedniki z alfabetu łacińskiego oraz losowa zamiana dwóch znaków. Znacznie pogorszyło to jakość embeddingów. Natomiast, mniejszą zmianę odnotowałem w embeddingach bezkontekstowych.


**Wyniki:**
papuga original # avg: 0.745648, max: 0.665764
BERT original # avg: 0.65476, max: 0.64039, first: 0.648844
papuga deformed # avg: 0.53862, max: 0.518726
BERT deformed # avg: 0.535188

## **Zadanie 2:**

**Cel:** 

**Rozwiązanie:**

## **Zadanie 3:**

**Cel:** 

**Rozwiązanie:**

## **Zadanie 4:**

**Cel:** 

**Rozwiązanie:**