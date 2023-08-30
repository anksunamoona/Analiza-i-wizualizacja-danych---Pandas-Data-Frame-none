import numpy as np 
import pandas as pd
import sklearn 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.preprocessing import StandardScaler
ss= StandardScaler()
sklearn.__version__
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.graphics.gofplots import qqplot
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
plt.style.use('ggplot')
from scipy.stats import chi2_contingency
import tkinter  
from tkinter import Tk, Label, Button
from tkinter import font as tkfont


# Załadowanie pliku, zawierającego zbiór baz danych użytkowników serwisu NETFLIX. Zbiór pochodzi ze strony Kaggle i jest publicznie dostępny do użytku.
print("\n\n\tZbiór danych bazy użytkowników serwisu NETFLIX\n")
df = pd.read_csv('Netflix.csv',index_col= 0)
print(df)

age_column = df['Age']


# Selekcja cech ze zbioru bazy danych użytkowników serwisu NETFLIX. 
print("\n\n\tSelekcja cech oraz iteracja po kolumnach\n")
print(df.columns)
for col in df.columns:
    print(col)


# Przedstawienie danych statystycznych zbioru bazy danych użytkowników serwisu NETFLIX.
print("\n\n\tSzczegóły statystyczne\n")
print(df.describe()) 


# Podstawowe informacje o zmiennych, liczbie wartości różnych od wartości "null", typie danych w każdej kolumnie oraz wykorzystaniu pamięci.
print("\n\n\tInformacje o zmiennych, liczbie wartości różnych od wartości 'null', typie danych każdej kolumny i wykorzystaniu pamięci\n")
print(df.info())


# Zliczanie wartości "null", występujących w zbiorze bazy danych użytkowników serwisu NETFLIX.
print("\n\n\tZliczanie wartości null\n")
print(df.isnull().sum())


# Zliczanie wartości unikalnych, występujących w zbiorze bazy danych użytkowników serwisu NETFLIX.
print("\n\n\tZliczanie unikalnych wartości\n")
df['Subscription Type'].value_counts()
df['Monthly Revenue'].value_counts()
df['Join Date'].value_counts()
df['Last Payment Date'].value_counts()
df['Country'].value_counts()
df['Age'].value_counts()
df['Gender'].value_counts()
df['Device'].value_counts()
df['Plan Duration'].value_counts()


# Obliczanie mody dla kolumn zbioru bazy danych użytkowników serwisu NETFLIX.
print('\n\n\t Mediana kolumny Age\n')
median_Age = df['Age'].median()
print("Median of column Age:", median_Age)

print('\n\n\tModa kolumn zbioru bazy danych użytkowników serwisu NETFLIX\n')
mode_Subscription = df['Subscription Type'].mode()
print("\nMode of column Subscription Type")
print(mode_Subscription)

mode_Revenue = df['Monthly Revenue'].mode()
print("\nMode of column Monthly Revenue:") 
print(mode_Revenue)

mode_Date = df['Join Date'].mode()
print("\nMode of column Join Date:")
print(mode_Date)

mode_Last_Date = df['Last Payment Date'].mode()
print("\nMode of column Last Payment Date:")
print(mode_Last_Date)

mode_Country = df['Country'].mode()
print("\nMode of column Country")
print(mode_Country)

mode_Age = df['Age'].mode()
print("\nMode of column Age:")
print(mode_Age)

mode_Gender = df['Gender'].mode()
print("\nMode of column Gender")
print(mode_Gender)

mode_Device = df['Device'].mode()
print("\nMode of column Device")
print(mode_Device)

mode_Plan = df['Plan Duration'].mode()
print("\nMode of column Plan Duration")
print(mode_Plan)


# Obliczanie wariancji kolumny 'Age', zawierającej dane o wieku użytkowników serwisu NETFLIX.
print('\n\n\tWariancja kolumny Age\n')
var_Age = df['Age'].var()
print("Variance of column Age:", var_Age)


# Obliczanie wariancji kolumny 'Age', zawierającej dane o wieku użytkowników serwisu NETFLIX.
print('\n\n\tOdchylenie standardowe kolumny Age\n')
std = age_column.std()
print("Standard deviation:")
print(std)


#Obliczanie kwantyli kolumny 'Age', zawierającej dane o wieku użytkowników serwisu NETFLIX.
print("\n\n\tObliczanie kwantyli kolumny Age\n")

# Obliczanie mediany
median = age_column.quantile(0.5)
print("Median:", median)

# Obliczanie kwantylu 0.25
p25 = age_column.quantile(0.25)
print("\n25th percentile:", p25)

# Obliczanie kwantylu 0.75
p75 = age_column.quantile(0.75)
print("\n75th percentile:", p75)


# Obliczanie skośności kolumny 'Age', zawierającej dane o wieku użytkowników serwisu NETFLIX.
print("\n\n\tObliczanie skośności kolumny Age\n")
skewness = age_column.skew()
print("Skewness:", skewness)
# Wynik różny od zera świadczy o tym, że rozkład jest skośny, asymetryczny. W naszym przypadku wynik jest dodatni, co świadczy o tym, że nasz rozkład jest skońcy w prawo.


# Obliczanie kurtozy kolumny "Age", zawierającej dane o wieku użytkowników serwisu NETFLIX.
print("\n\n\tObliczanie kurtozy kolumny Age\n")
kurtosis = age_column.kurtosis()
print("Kurtosis:", kurtosis)
# Wynik ujemny świadczy o tym, że pik jest bardziej "płaski", czyli rozkład prawdopodobieństwa jest niższy względem rozkładu normalne, natomiast wynik dodatni świadczy, 
# o tym że pik jest ostrzej zakończony, czyli rozkład prawdopodobieństwa jest wyższy względem rozkładu normalnego.
# W naszym przypadku wynik jest ujemny - rozkład prawdopodobieństwa jest niższy.


#Tworzenie grup wiekowych "Young", "Middle-Aged" oraz "Senior" w celu przedstawienia na wykresie nr 1
def categorize_age(Age):
    if Age < 30:
        return 'Young (x < 30)'
    elif Age >= 30 and Age < 50:
        return 'Middle-aged (50 < x > 30)'
    else:
        return 'Senior (x > 50)'

df['Age group'] = df['Age'].apply(categorize_age)


# Tworzenie grup, dzielących użytkowników według wieku:
#  - użytkownicy do 30 roku życia,
#  - użytkownicy, między 30, a 35 rokiem życia,
#  - użytkownicy, między 35, a 40 rokiem życia,
#  - użytkownicy, między 40, a 45 rokiem życia,
#  - użytkownicy, między 45, a 50 rokiem życia,
#  - użytkownicy powyżej 50 roku życia.
#  w celu przejrzystego przedstawienia zależności, między wiekiem, a urządzeniem, z którego korzystają użytkownicy serwisu NETFLIX na wykresie nr 2
def categorize_age2(Age):
    if Age < 30:
        return 'Użytkownicy do 30 roku życia'
    elif Age >= 30 and Age < 35:
        return 'Użytkownicy między 30,\n a 35 rokiem życia'
    elif Age >= 35 and Age < 40:
        return 'Użytkownicy między 35,\n a 40 rokiem życia'
    elif Age >= 40 and Age < 45:
        return 'Użytkownicy między 40,\n a 45 rokiem życia'
    elif Age >= 45 and Age < 50:
        return 'Użytkownicy między 45,\n a 50 rokiem życia'
    else:
        return 'Użytkownicy powyżej 50 roku życia'
    
df['Users Age'] = df['Age'].apply(categorize_age2)


# Przeprowadzenie testu chi-kwadrat na podstawie danych z kolumny Age oraz Subscription Type.
contingency_table = pd.crosstab(df['Age'], df['Subscription Type'])
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"Statystyka testu chi-kwadrat: {chi2}")
print(f"P-wartość: {p}")
print(f"Stopnie swobody: {dof}")
print("Oczekiwane wartości:")
print(expected)

# Przeprowadzenie testu chi-kwadrat na podstawie danych z kolumny Age oraz Country.
contingency_table = pd.crosstab(df['Age'], df['Country'])
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"Statystyka testu chi-kwadrat: {chi2}")
print(f"P-wartość: {p}")
print(f"Stopnie swobody: {dof}")
print("Oczekiwane wartości:")
print(expected)

# Wizualizacja danych ze zbioru bazy danych użytkowników serwisu NETFLIX.

while True:
    print("\n\n\tDostępne wykresu do wyboru:\n")
    print("1. Wykres bazy użytkowników serwisu NETFLIX z podziałem na płeć oraz grupę wiekową.")
    print("\n2. Wykres zależności, między wiekiem użytkowników serwisu NETFLIX, a urządzeniem, z którego korzystają.")
    print("\n3. Wykres zależności, między krajem zamieszkania użytkowników, a wykorzystywanym planem subskrypcji użytkowników serwisu NETFLIX.")
    print("\n4. Wykres zależności, między krajem zamieszkania użytkowników, a urządzeniem, z którego korzystają użytkownicy serwisu NETFLIX.")
    print("\n5. Wykres, przedstawiający histogram kolumny Age.")
    print("\n6. Wykres przedstawiający rozkład normalny.")
    print("\n7. Wykres skośności.")
    print("\n8. Wykres kurtozy.")
    print("\n9. Wykres słupkowy wyników testu statystycznego chi-kwadrat.")
    print("\n10. Wykres typu heatmap wyników testu chi-kwadrat.")
    print("\n11. Wykres typu heatmap wyników testu chi-kwadrat.")
    print("\n12. Zakończ przegląd wykresów.")
    print("\n13. Autorki projektu.")
    choice = input("\nWybierz opcję (1/13): ")

    if choice == '1':
    # Wykres nr 1 - przedstawienie bazy użytkowników serwisu NETFLIX z podziałem na płeć oraz grupę wiekową.
        plt.figure(figsize=(17,8))
        sns.countplot(x='Age group', hue='Gender', data = df, palette="Accent", saturation = 1)
        plt.show()
    elif choice == '2':
    # Wykres nr 2 - przedstawienie zależności, między wiekiem użytkowników serwisu NETFLIX, a urządzeniem, z którego korzystają.
        plt.figure(figsize=(17,8))
        sns.countplot(x='Users Age', hue='Device', data = df, palette="Set1", saturation = 1)
        plt.show()
    elif choice == '3':
    # Wykres nr 3 - przedstawienie zależności, między krajem zamieszkania użytkowników, a wykorzystywanym planem subskrypcji użytkowników serwisu NETFLIX.
        plt.figure(figsize=(17,8))
        sns.countplot(x='Country', hue='Subscription Type', data = df, palette="plasma", saturation = 1)
        plt.show()
    elif choice == '4':
    # Wykres nr 4 - przedstawienie zależności, między krajem zamieszkania użytkowników, a urządzeniem, z którego korzystają użytkownicy serwisu NETFLIX.
        plt.figure(figsize=(17,8))
        sns.countplot(x='Country', hue='Device', data = df, palette="Set1", saturation = 1)
        plt.show()
    elif choice == '5':
    # Wykres nr 5 - histogram kolumny "Age", czyli wieku użytkowników serwisu NETFLIX.
        age_column.hist(bins=100, figsize=(17,8))
        plt.xlabel("Wiek użytkowników serwisu Netflix")
        plt.ylabel("Ilość użytkowników serwisu Netflix")
        plt.title("Histogram kolumny Age, zawierającej dane o wieku użytkowników seriwus NETFLIX")
        plt.show()
    elif choice == '6': 
    # Wykres nr 6 - przedstawia wykres rozkładu normalnego.
        fig, ax = plt.subplots(figsize=(10, 8))
        qqplot(age_column, line='s', ax=ax)
        ax.set_xlabel('Rozkład teoretyczny')
        ax.set_ylabel('Rozkład zbioru danych')
        ax.set_title('Wykres Q-Q dla rozkładu normalnego')
        plt.show()
    elif choice == '7': 
    # Wykres nr 7 - wykres skośności rozkładu.
        plt.hist(age_column.skew(), bins=30, alpha=0.7, density=True, color='blue', label='Dane')
        plt.axvline(np.mean(age_column.skew()), color='red', linestyle='dashed', linewidth=1, label='Średnia')
        plt.title(f'Wykres skośności\nSkośność: {skewness:.2f}')
        plt.xlabel('Wartości')
        plt.ylabel('Częstotliwość')
        plt.legend()
        plt.show()
    elif choice == '8':
    # Wykres nr 8 - wykres kurtozy rozkładu.    
        plt.hist(age_column.kurtosis(), bins=30, alpha=0.7, density=True, color='blue', label='Dane')
        plt.axvline(np.mean(age_column.kurtosis()), color='red', linestyle='dashed', linewidth=1, label='Średnia')
        plt.title(f'Wykres kurtozy\nKurtoza: {kurtosis:.2f}')
        plt.xlabel('Wartości')
        plt.ylabel('Częstotliwość')
        plt.legend()
        plt.show()
    elif choice == '9':
    # Przedstawienie wyników testu chi-kwadrat, przy pomocy wykresu słupkowego. Ten test został wybrany ze względu na charakter typu danych 
    # w użytym przez nas zbiorze - znajdują się w nim typy mieszane (int, string).
        df.groupby('Age')['Subscription Type'].value_counts().unstack().plot(kind='bar', stacked=True, figsize=(17,8))
        plt.title('Wynik w zależności od kategorii')
        plt.xlabel('Wiek')
        plt.ylabel('Typ subskrypcji')
        plt.show()
    elif choice == '10':
    # Przedstawienie wyników testu chi-kwadrat, przy pomocy heatmapy.
        plt.figure(figsize=(17, 8))
        sns.heatmap(contingency_table, annot=True, cmap='YlGnBu', fmt='d')
        plt.title('Tabela kontyngencji')
        plt.xlabel('Plan Subskrypcji')
        plt.ylabel('Wiek')
        plt.show()
    elif choice == '11':
    # Przedstawienie wyników testu chi-kwadrat, przy pomocy heatmapy.
        plt.figure(figsize=(17, 8))
        sns.heatmap(contingency_table, annot=True, cmap='YlGnBu', fmt='d')
        plt.title('Tabela kontyngencji')
        plt.xlabel('Kraj')
        plt.ylabel('Wiek')
        plt.show()
    elif choice == '12':
        print("Zakończ przegląd wykresów.")
        break
    elif choice == '13':
    #Wyskakujące okno z autorkami 
        root = tkinter.Tk()
        root.geometry("300x300")
        appHighlightFont = tkfont.Font(family='Helvetica', size=14, weight='bold')
        text_label = Label(root, text="\n\n\nProjekt przygotowały:\n\nAnna Vezdenetska\n\nAleksandra Kot", font=appHighlightFont)
        click_button = Button(root, text="OK", width=8)
        click_button.pack(side = tkinter.BOTTOM)
        text_label.pack(pady = 5)
        click_button.config(command = root.destroy)
        root.mainloop()
    else:
        print("Nieprawidłowy wybór.")




