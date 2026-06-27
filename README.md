# NCG_jako_solver_regresji_logistycznej-_por-wnanie_z_GD_i_Newtonem

Projekt 6 — NCG jako solver regresji logistycznej
Projekt został wykonany w ramach zajęć z metod optymalizacji. Celem projektu jest zastosowanie metod optymalizacji ciągłej do minimalizacji funkcji straty logistycznej oraz porównanie skuteczności kilku algorytmów na problemie klasyfikacji binarnej.

W projekcie wykorzystano zbiór breast_cancer dostępny w bibliotece sklearn.datasets.

Autorzy:

•	Alicja Pakulska

•	Krystyna Trzusło

# Cel projektu
Celem projektu jest rozwiązanie problemu regresji logistycznej trzema metodami optymalizacji:
1.	Gradient Descent ze stałym krokiem,
2.	NCG-PRP+ z backtrackingiem Armijo,
3.	Newton-Raphson z backtracking line search.
Dla każdej metody porównano:
•	liczbę iteracji,
•	czas działania,
•	końcową wartość funkcji straty,
•	accuracy na zbiorze testowym.
Projekt zawiera również rozszerzenia, takie jak porównanie wariantów metody NCG, analiza wpływu regularyzacji L2, porównanie kosztu pamięciowego metod oraz porównanie własnej implementacji z biblioteką referencyjną.

# Struktura projektu

project/

├── README.md

├── requirements.txt

├── notebooks/

│   └── Projekt_Metody.ipynb

├── tests/

│   └── test_validation.py

├── report.pdf

└── presentation.pdf

# Opis plików:

•	README.md — instrukcja uruchomienia projektu,

•	requirements.txt — lista wymaganych bibliotek,

•	notebooks/Projekt_Metody.ipynb — notebook demonstracyjny z implementacją metod, wynikami i wykresami,

•	tests/test_validation.py — testy lub przykłady walidacyjne,

•	report.pdf — raport techniczny,

•	presentation.pdf — krótka prezentacja wyników.

# Wymagania

Do uruchomienia projektu wymagane są biblioteki:

numpy

matplotlib

scikit-learn

W przypadku uruchamiania projektu lokalnie można zainstalować biblioteki poleceniem:

pip install -r requirements.txt

# Uruchomienie w Google Colab

Projekt był przygotowywany z myślą o uruchomieniu w Google Colab.

# Instrukcja uruchomienia:

1.	Otworzyć plik Projekt_Metody.ipynb w Google Colab.

2.	Uruchomić wszystkie komórki przez Runtime -> Run all.

3.	Sprawdzić tabelę wyników dla metod GD, NCG-PRP+ i Newton-Raphson.

4.	Przeanalizować wykresy zbieżności oraz końcowe wnioski.

# Uruchomienie lokalne

Aby uruchomić projekt lokalnie, należy najpierw sklonować repozytorium lub pobrać pliki projektu.

Następnie można utworzyć środowisko wirtualne:

python -m venv venv

Aktywacja środowiska w systemie Windows:

venv\Scripts\activate

Aktywacja środowiska w systemie Linux/macOS:

source venv/bin/activate

Instalacja wymaganych bibliotek:

pip install -r requirements.txt

Po instalacji bibliotek można uruchomić notebook:

jupyter notebook notebooks/Projekt_Metody.ipynb

# Dane

W projekcie wykorzystano zbiór breast_cancer z biblioteki sklearn.datasets.

Zbiór zawiera:

•	569 obserwacji,

•	30 cech,

•	dwie klasy decyzyjne.

Dane zostały podzielone na zbiór treningowy i testowy. Przed treningiem wykonano standaryzację cech, ponieważ metody gradientowe są wrażliwe na skalę zmiennych. Etykiety klas zostały przekształcone z postaci {0, 1} do postaci {-1, 1}, co ułatwia zapis funkcji straty logistycznej.

# Zaimplementowane metody

Gradient Descent

Gradient Descent jest metodą pierwszego rzędu. W każdej iteracji obliczany jest gradient funkcji straty, a następnie wykonywany jest krok w kierunku przeciwnym do gradientu. W projekcie zastosowano stałą długość kroku.

NCG-PRP+

NCG-PRP+ to nieliniowa metoda gradientów sprzężonych. Metoda wykorzystuje aktualny gradient oraz informację z poprzednich iteracji do wyznaczania kolejnych kierunków poszukiwań. Długość kroku dobierana jest za pomocą backtrackingu Armijo.

Newton-Raphson

Metoda Newtona-Raphsona wykorzystuje gradient oraz hesjan funkcji straty. Kierunek Newtona wyznaczany jest przez rozwiązanie układu równań liniowych. W implementacji użyto np.linalg.solve, bez jawnego obliczania odwrotności macierzy Hessego.

# Wyniki

Dla każdej metody zapisano:

•	liczbę iteracji,

•	czas CPU,

•	końcową wartość funkcji straty,

•	accuracy na zbiorze testowym.

Wyniki znajdują się w notebooku demonstracyjnym oraz w raporcie technicznym. Notebook zawiera także wykresy pokazujące zmianę wartości funkcji straty i normy gradientu w kolejnych iteracjach.

# Walidacja implementacji

W projekcie przygotowano przykłady walidacyjne sprawdzające poprawność implementacji. Sprawdzono między innymi:

•	czy funkcja straty zwraca poprawną wartość dla punktu startowego,

•	czy gradient ma poprawny wymiar,

•	czy hesjan ma poprawny wymiar,

•	czy po małym kroku w kierunku przeciwnym do gradientu wartość funkcji straty maleje.

# Porównanie z biblioteką referencyjną

Projekt zawiera porównanie własnej implementacji z biblioteką referencyjną sklearn.linear_model.LogisticRegression. Pozwala to sprawdzić, czy własna implementacja daje wyniki porównywalne z gotowym rozwiązaniem bibliotecznym.

# Raport i prezentacja
Szczegółowy opis metod, wzory matematyczne, wyniki eksperymentów oraz wnioski znajdują się w pliku:

report.pdf

Krótka prezentacja wyników znajduje się w pliku:

presentation.pdf

