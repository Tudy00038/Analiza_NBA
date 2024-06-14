
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation


#Citim setul de date
df = pd.read_csv("./team_scoring.csv", encoding='latin1')

#Aflam structura setului de date si informatii despre acesta
print(df.info())
print(df.describe())

#Stergem valorile nule
df.dropna(inplace=True)

#Înlocuim valorile NaN cu 0************
df.fillna(0, inplace=True)

#Verificam daca exista valori duplicate si in cazul in care exista le stergem(in caszul nostu avem 0 valori duplicate)
valori_duplicate = df.duplicated()
print("Numar de valori duplicate:", valori_duplicate.sum())

#Numarul total de inregistrari
numar_inregistrari = df["gameid"]
print ("Numar inregistrari: ", numar_inregistrari.count())

# Convertim coloana 'data' în formatul date
df['date'] = pd.to_datetime(df['date'], errors='coerce', format='%Y-%m-%d')

# Extragem doar componenta de dată
df['date'] = df['date'].dt.date

###############################################################################

#medii / cvartile/deviatie standard / mediana pt var. numerice

# Obținerea descrierii datelor
summary_stats = df.describe()

###############################################################################

#Analiza exploratorie

# Filtrează pentru sezonul 2023 și tipul de meci "playoff"
playoff_2023 = df[(df['season'] == 2023) & (df['type'] == 'playoff')]

# Filtrează meciurile câștigate
victories = playoff_2023[playoff_2023['win'] == 1]

# Numără câte victorii are fiecare echipă
team_victories = victories.groupby('team')['win'].count()

# Găsește echipa cu cele mai multe victorii
top_team = team_victories.idxmax()  # Numele echipei cu cele mai multe victorii
top_team_victories = team_victories.max()  # Numărul maxim de victorii

print(f'Echipa cu cele mai multe victorii în playoff în sezonul 2023 este: {top_team} cu {top_team_victories} victorii.')


# Extrage datele doar pentru echipa cu cele mai multe victorii
top_team_data = playoff_2023[playoff_2023['team'] == top_team]

# Vizualizează câteva rânduri pentru a înțelege structura datelor
print(top_team_data.head())

# Distribuția punctelor
distribution = top_team_data[['%PTS 2PT', '%PTS 3PT', '%PTS FT', '%PTS FBPS', '%PTS OFF TO', '%PTS PITP']].mean()

print("Distribuția medie a punctelor pentru echipa cu cele mai multe victorii:")
print(distribution)

# Proporția de field goal attempts 2PT vs 3PT
fga_2pt = top_team_data['%FGA 2PT'].mean()
fga_3pt = top_team_data['%FGA 3PT'].mean()

print("Proporția de field goal attempts:")
print(f"2PT: {fga_2pt:.2f}%")
print(f"3PT: {fga_3pt:.2f}%")

# Proporția de field goals asistate și neasistate
assist_stats = top_team_data[['2FGM %AST', '2FGM %UAST', '3FGM %AST', '3FGM %UAST']].mean()

print("Proporția de field goals asistate și neasistate:")
print(assist_stats)

# Numărul total de victorii pentru echipa
total_wins = top_team_data['win'].sum()

print(f"Totalul de victorii pentru echipa {top_team}: {total_wins}")

# Vizualizează distribuția punctelor
distribution.plot(kind='bar', title="Distribuția punctelor pentru echipa cu cele mai multe victorii")
plt.show()

# Vizualizează proporția de field goal attempts
plt.pie([fga_2pt, fga_3pt], labels=['2PT', '3PT'], autopct='%1.1f%%', startangle=90)
plt.title("Proporția de field goal attempts pentru echipa cu cele mai multe victorii")
plt.show()

# Vizualizează proporția de field goals asistate și neasistate
assist_stats.plot(kind='bar', title="Proporția de field goals asistate și neasistate")
plt.show()

######################################

# Filtrează pentru sezonul 2023 și tipul de meci "playoff"
playoff_1997 = df[(df['season'] == 1997) & (df['type'] == 'playoff')]

# Filtrează meciurile câștigate
victories1997 = playoff_1997[playoff_1997['win'] == 1]

# Numără câte victorii are fiecare echipă
team_victories1997 = victories1997.groupby('team')['win'].count()

# Găsește echipa cu cele mai multe victorii
top_team1997 = team_victories1997.idxmax()  # Numele echipei cu cele mai multe victorii
top_team_victories1997 = team_victories1997.max()  # Numărul maxim de victorii

print(f'Echipa cu cele mai multe victorii în playoff în sezonul 1997 este: {top_team1997} cu {top_team_victories1997} victorii.')


# Extrage datele doar pentru echipa cu cele mai multe victorii
top_team_data1997 = playoff_1997[playoff_1997['team'] == top_team1997]

# Vizualizează câteva rânduri pentru a înțelege structura datelor
print(top_team_data1997.head())

# Distribuția punctelor
distribution1997 = top_team_data1997[['%PTS 2PT', '%PTS 3PT', '%PTS FT', '%PTS FBPS', '%PTS OFF TO', '%PTS PITP']].mean()

print("Distribuția medie a punctelor pentru echipa cu cele mai multe victorii:")
print(distribution1997)

# Proporția de field goal attempts 2PT vs 3PT
fga_2pt_1997 = top_team_data1997['%FGA 2PT'].mean()
fga_3pt_1997 = top_team_data1997['%FGA 3PT'].mean()

print("Proporția de field goal attempts:")
print(f"2PT: {fga_2pt_1997:.2f}%")
print(f"3PT: {fga_2pt_1997:.2f}%")

# Proporția de field goals asistate și neasistate
assist_stats1997 = top_team_data1997[['2FGM %AST', '2FGM %UAST', '3FGM %AST', '3FGM %UAST']].mean()

print("Proporția de field goals asistate și neasistate:")
print(assist_stats1997)

# Numărul total de victorii pentru echipa
total_wins1997 = top_team_data1997['win'].sum()

print(f"Totalul de victorii pentru echipa {top_team1997}: {total_wins1997}")

# Vizualizează distribuția punctelor
distribution1997.plot(kind='bar', title="Distribuția punctelor pentru echipa cu cele mai multe victorii", color="green")
plt.show()

# Vizualizează proporția de field goal attempts
plt.pie([fga_2pt_1997, fga_3pt_1997], labels=['2PT', '3PT'], autopct='%1.1f%%', startangle=90, colors=["green","red"])
plt.title("Proporția de field goal attempts pentru echipa cu cele mai multe victorii")
plt.show()

# Vizualizează proporția de field goals asistate și neasistate
assist_stats1997.plot(kind='bar', title="Proporția de field goals asistate și neasistate", color="green")
plt.show()

####################################

# Filtrează datele după tipul de meci (playoff, playin, regular)
playoff_data = df[df['type'] == 'playoff']
playin_data = df[df['type'] == 'playin']
regular_data = df[df['type'] == 'regular']

# Numără câte victorii are fiecare echipă în fiecare tip de meci
playoff_victories = playoff_data[playoff_data['win'] == 1].groupby('team')['win'].count()
playin_victories = playin_data[playin_data['win'] == 1].groupby('team')['win'].count()
regular_victories = regular_data[regular_data['win'] == 1].groupby('team')['win'].count()

# Echipele cu cele mai multe victorii în fiecare categorie
top_playoff_team = playoff_victories.idxmax()  # Numele echipei cu cele mai multe victorii în playoff
top_playin_team = playin_victories.idxmax()  # Numele echipei cu cele mai multe victorii în playin
top_regular_team = regular_victories.idxmax()  # Numele echipei cu cele mai multe victorii în sezonul regulat

# Creează un grafic pentru victoriile în playoff
playoff_victories.plot(kind='bar', title="Victoriile în playoff", color="orange")
plt.xlabel("Echipe")
plt.ylabel("Număr de Victorii")
plt.show()

# Creează un grafic pentru victoriile în playin
playin_victories.plot(kind='bar', title="Victoriile în playin", color="orange")
plt.xlabel("Echipe")
plt.ylabel("Număr de Victorii")
plt.show()

# Creează un grafic pentru victoriile în sezonul regulat
regular_victories.plot(kind='bar', title="Victoriile în sezonul regulat", color="orange")
plt.xlabel("Echipe")
plt.ylabel("Număr de Victorii")
plt.show()

print(f'Echipa cu cele mai multe victorii în playoff: {top_playoff_team}')
print(f'Echipa cu cele mai multe victorii în playin: {top_playin_team}')
print(f'Echipa cu cele mai multe victorii în sezonul regulat: {top_regular_team}')


# Filtrăm doar meciurile de playoff
df_playoff = df[df['type'] == 'playoff']

# Grupăm și numărăm câștigurile pe echipă și sezon
df_grouped = df_playoff.groupby(['season', 'team'])['win'].sum().reset_index()

# Sortăm valorile după sezon și numărul de câștiguri în ordine descrescătoare
df_sorted = df_grouped.sort_values(['season', 'win'], ascending=[True, False])

# Alegem câștigătoarea pentru fiecare sezon
df_winners = df_sorted.drop_duplicates(subset=['season'], keep='first')

print("Echipele câștigătoare din fiecare sezon:")
print(df_winners)

# Creăm un grafic de tip bar pentru a afișa câștigurile echipelor câștigătoare în fiecare sezon
plt.figure(figsize=(10, 6))
sns.barplot(x='season', y='win', data=df_winners, hue='team', dodge=False)
plt.title("Echipele câștigătoare din fiecare sezon (playoff)")
plt.xlabel("Sezon")
plt.ylabel("Număr de câștiguri")
plt.show()

# Grupăm după echipă pentru a număra aparițiile
df_team_count = df_winners.groupby('team')['season'].count().reset_index()

# Sortăm pentru a obține echipele cu cele mai multe apariții
df_team_count = df_team_count.sort_values(by='season', ascending=False)

# Redenumim coloana 'season' pentru a indica că aceasta este numărul de apariții
df_team_count = df_team_count.rename(columns={'season': 'appearances'})

print("Echipele cu cele mai multe apariții ca câștigătoare:")
print(df_team_count)
#=> ECHIPA CU CELE MIA MULTE TITLURI CASTIGATE din 97 pana 23

# Grafic de tip bar pentru numărul de apariții ca câștigătoare
plt.figure(figsize=(10, 6))
sns.barplot(x='team', y='appearances', data=df_team_count, order=df_team_count['team'], color="salmon")
plt.title("Echipele cu cele mai multe apariții ca câștigătoare")
plt.xlabel("Echipa")
plt.ylabel("Număr de apariții")
plt.show()

##########################

# Sortăm după sezon pentru a evalua câștigurile consecutive
df_winners = df_winners.sort_values(by='season')

# Initializăm contorii pentru echipele câștigătoare consecutive
max_consecutive = 0
current_consecutive = 0
current_team = None
current_seasons = []
max_seasons = []
best_team = None

# Iterăm peste fiecare rând și contorizăm câștigurile consecutive
for _, row in df_winners.iterrows():
    team = row['team']
    season = row['season']

    if team == current_team:
        # Dacă echipa este aceeași, creștem contorul și adăugăm sezonul
        current_consecutive += 1
        current_seasons.append(season)
    else:
        # Dacă echipa s-a schimbat, verificăm pentru maxim și resetăm
        if current_consecutive > max_consecutive:
            max_consecutive = current_consecutive
            best_team = current_team
            max_seasons = current_seasons[:]

        # Resetăm pentru noua echipă
        current_team = team
        current_consecutive = 1  # începem o nouă serie
        current_seasons = [season]  # resetăm seria de sezoane

# Asigurăm că verificăm ultima serie pentru maxim
if current_consecutive > max_consecutive:
    max_consecutive = current_consecutive
    best_team = current_team
    max_seasons = current_seasons[:]

print(f"Echipa cu cele mai multe câștiguri consecutive este: {best_team} cu {max_consecutive} câștiguri consecutive.")
print(f"Seria de sezoane pentru aceste câștiguri: {max_seasons}")


# Sortăm după sezon pentru a evalua câștigurile consecutive
df_winners = df_winners.sort_values(by='season')

# Lista pentru a păstra echipele cu cel puțin două câștiguri consecutive
consecutive_teams = []
current_team = None
current_consecutive = 0

# Parcurgem datele pentru a identifica câștigurile consecutive
for _, row in df_winners.iterrows():
    team = row['team']

    if team == current_team:
        # Dacă echipa este aceeași, creștem contorul
        current_consecutive += 1
    else:
        # Dacă echipa s-a schimbat, verificăm dacă au fost cel puțin două consecutive
        if current_consecutive >= 2:
            consecutive_teams.append({
                'team': current_team,
                'consecutive_wins': current_consecutive,
                'start_season': row['season'] - (current_consecutive - 1),
                'end_season': row['season']
            })

        # Resetăm pentru noua echipă
        current_team = team
        current_consecutive = 1

# Verificăm ultima serie pentru cel puțin două consecutive
if current_consecutive >= 2:
    consecutive_teams.append({
        'team': current_team,
        'consecutive_wins': current_consecutive,
        'start_season': df_winners['season'].iloc[-1] - (current_consecutive - 1),
        'end_season': df_winners['season'].iloc[-1]
    })

# Creăm un dataframe cu echipele care au cel puțin două câștiguri consecutive
df_consecutive = pd.DataFrame(consecutive_teams)

print("Echipele cu cel puțin două câștiguri consecutive:")
print(df_consecutive)


#Grupăm datele după echipă și calculăm suma pentru fiecare categorie
total_scores_by_team = df.groupby('team')[['%FGA 2PT', '%FGA 3PT', '%PTS 2PT', '%PTS 2PT MR', '%PTS 3PT', '%PTS FBPS', '%PTS FT', '%PTS OFF TO', '%PTS PITP', '2FGM %AST', '2FGM %UAST', '3FGM %AST', '3FGM %UAST', 'FGM %AST', 'FGM %UAST']].sum()

#Determinăm echipa cu cele mai mari sume pentru fiecare categorie
max_scores_by_category = total_scores_by_team.idxmax()

print("Echipa cu cele mai multe coșuri înscrise în total pentru fiecare categorie:")
print(max_scores_by_category)


# Determinăm echipele câștigătoare consecutive
# Folosește codul anterior pentru a crea df_consecutive
consecutive_teams = []
current_team = None
current_consecutive = 0

for _, row in df_winners.iterrows():
    team = row['team']

    if team == current_team:
        current_consecutive += 1
    else:
        if current_consecutive >= 2:
            consecutive_teams.append(current_team)
        current_team = team
        current_consecutive = 1

if current_consecutive >= 2:
    consecutive_teams.append(current_team)

df_consecutive = pd.DataFrame({
    'team': consecutive_teams
})

# Obținem statisticile pentru punctele de două și trei puncte
team_stats = df.groupby('team').agg({
    '%FGA 2PT': 'mean',
    '%FGA 3PT': 'mean',
    '%PTS 2PT': 'mean',
    '%PTS 3PT': 'mean'
}).reset_index()

# Comparăm punctele pentru echipele câștigătoare consecutive
df_comparison = team_stats[team_stats['team'].isin(df_consecutive['team'])]

# Determinăm dacă echipele au mai multe puncte din aruncări de două puncte decât din trei puncte
df_comparison['more_2PT_than_3PT'] = df_comparison['%PTS 2PT'] > df_comparison['%PTS 3PT']

# Creează un grafic pentru a compara punctele din aruncări de două și trei puncte
plt.figure(figsize=(10, 6))
df_comparison.plot(x='team', y=['%PTS 2PT', '%PTS 3PT'], kind='bar', title="Compararea punctelor din aruncări de două și trei puncte", color=['blue', 'orange'], ax=plt.gca())
plt.xlabel("Echipă")
plt.ylabel("Procentul de puncte")
plt.show()


#
# Grupăm după echipă și numărăm de câte ori apare fiecare echipă
team_appearances = df_winners.groupby('team')['season'].count()

# Convertim într-un DataFrame și sortăm descrescător după numărul de apariții
team_appearances_df = pd.DataFrame(team_appearances).reset_index()
team_appearances_df.columns = ['team', 'appearances']
team_appearances_df = team_appearances_df.sort_values(by='appearances', ascending=False)

# Afișează primele 10 echipe cu cele mai multe apariții
print(team_appearances_df.head(10))

# Creează un grafic barplot pentru a vizualiza echipele cu cele mai multe apariții
sns.barplot(data=team_appearances_df.head(10), x='appearances', y='team', palette='viridis')
plt.title("Echipele care au cele mia multe campionate castigate")
plt.xlabel("Număr de Apariții")
plt.ylabel("Echipe")
plt.show()




# Grupează după echipă și numără de câte ori apare fiecare echipă
team_counts = df_winners.groupby('team')['season'].count()

# Identifică echipa care apare de cele mai multe ori
most_common_team = team_counts.idxmax()  # Echipa cu cele mai multe apariții
most_common_count = team_counts.max()  # Numărul maxim de apariții

print(f'Echipa care a apărut de cele mai multe ori este: {most_common_team}, cu {most_common_count} apariții.')







# Obținem statisticile pentru punctele de două și trei puncte
team_stats_assists = df.groupby('team').agg({
    '2FGM %AST': 'mean',
    '3FGM %AST': 'mean',
    '2FGM %UAST': 'mean',
    '3FGM %UAST': 'mean',
    'FGM %AST': 'mean',
    'FGM %UAST': 'mean'
}).reset_index()

# Comparăm punctele pentru echipele câștigătoare consecutive
df_comparison_assists = team_stats_assists[team_stats_assists['team'].isin(df_consecutive['team'])]

# Definim figurile pentru grafic
plt.figure(figsize=(10, 6))

# Cream un grafic de bare pentru statisticile de asistență
sns.barplot(data=df_comparison_assists.melt(id_vars='team', 
                                            value_vars=['2FGM %AST', '3FGM %AST']),
            x='team', y='value', hue='variable')

# Titlu și etichete
plt.title("Comparare Asistențe pentru Aruncările de Două și Trei Puncte între Echipe")
plt.xlabel("Echipa")
plt.ylabel("Procentaj Asistențe")
plt.legend(title='Tip Aruncare')

# Afișare grafic
plt.show()





# Filtrează doar meciurile de tip "playoff"
playoff_data = df[df['type'] == 'playoff']

# Grupăm după echipă pentru a număra de câte ori au fost în playoffuri
team_playoff_counts = playoff_data.groupby('team')['gameid'].nunique()

# Creează un DataFrame pentru rezultatul final și ordonează descrescător
playoff_qualifications = pd.DataFrame(team_playoff_counts).reset_index()
playoff_qualifications.columns = ['team', 'playoff_count']
playoff_qualifications = playoff_qualifications.sort_values(by='playoff_count', ascending=False)

# Afișează primele rezultate pentru a vedea echipele care s-au calificat de cele mai multe ori
print(playoff_qualifications.head())

# Creează un grafic pentru a vizualiza echipele cu cele mai multe calificări în playoffuri
sns.barplot(data=playoff_qualifications.head(10), x='playoff_count', y='team', palette='viridis')  # Head(10) pentru primele 10 echipe
plt.title("Echipele care s-au calificat de cele mai multe ori în playoffuri")
plt.xlabel("Număr de Calificări")
plt.ylabel("Echipe")
plt.show()



#############################################################################################################################################################################################
coloane_categoriale = ['type']


df_encoded = pd.get_dummies(df, columns=coloane_categoriale, drop_first=True, dtype=int)

features = df_encoded.drop(['win','gameid','teamid','team','date','home','away'],axis=1)

target = df_encoded['win']

x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=0)
#################################################Matricea de corelatie################################################
correlation_matrix = df.corr(numeric_only=True)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

#Corelatie df_encoded
correlation_matrix_encoded = df_encoded.corr(numeric_only=True)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_encoded, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()


#Coeficient Pearson
# Lista pentru a stoca rezultatele
correlation_results = []

# Calculați și adăugați coeficientul de corelație Pearson în listă
for feature in features.columns:
    correlation, _ = pearsonr(features[feature], target)
    correlation_results.append({'Feature': feature, 'Pearson Correlation': correlation})

# Creare DataFrame din lista de rezultate
correlation_df = pd.DataFrame(correlation_results)

# Afișare DataFrame cu rezultatele
print(correlation_df)




# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(x_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(x_test)


# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))



from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score

clf = DecisionTreeClassifier()

# Define the parameter grid for grid search
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

# Perform grid search to find the best parameters
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(x_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Train the decision tree classifier with the best parameters
best_clf = DecisionTreeClassifier(**best_params)
best_clf.fit(x_train, y_train)

# Predict the response for test dataset
y_pred = best_clf.predict(x_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Perform k-fold cross-validation
cv_scores = cross_val_score(best_clf, x_train, y_train, cv=5)
print("Cross-Validation Mean Accuracy:", cv_scores.mean())

# Perform SMOTE oversampling if data is imbalanced
smote = SMOTE(random_state=42)
x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

# Train the decision tree classifier with resampled data
best_clf_resampled = DecisionTreeClassifier(**best_params)
best_clf_resampled.fit(x_train_resampled, y_train_resampled)

# Predict the response for test dataset with resampled data
y_pred_resampled = best_clf_resampled.predict(x_test)

# Evaluate accuracy with resampled data
accuracy_resampled = accuracy_score(y_test, y_pred_resampled)
print("Accuracy with SMOTE:", accuracy_resampled)

from sklearn.ensemble import AdaBoostClassifier

# Initialize AdaBoostClassifier with DecisionTreeClassifier as the base estimator
ada_clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(**best_params), n_estimators=50, learning_rate=1.0)

# Train AdaBoostClassifier
ada_clf.fit(x_train, y_train)

# Predict the response for test dataset
y_pred_boosted = ada_clf.predict(x_test)

# Evaluate accuracy
accuracy_boosted = accuracy_score(y_test, y_pred_boosted)
print("Boosted Decision Tree Accuracy:", accuracy_boosted)

from sklearn.ensemble import RandomForestClassifier

#Inițializează modelul Random Forest
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)  # Numărul de arbori este setat la 100, dar poți ajusta acest parametru

#Antrenează modelul pe datele de antrenare
random_forest.fit(x_train, y_train)

#Prezice etichetele pentru datele de test
y_pred_rf = random_forest.predict(x_test)

#Evaluarea performanței modelului Random Forest
accuracy_rf = metrics.accuracy_score(y_test, y_pred_rf)
print("Accuracy Random Forest:", accuracy_rf)


feature_importance = random_forest.feature_importances_

# Sortează caracteristicile în funcție de importanță
sorted_idx = feature_importance.argsort()

# Creează un grafic pentru importanța caracteristicilor
plt.figure(figsize=(10, 8))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), [x_train.columns[i] for i in sorted_idx])
plt.xlabel('Importanța caracteristicelor')
plt.title('Importanța caracteristicelor în Random Forest')
plt.show()


# Extrage numele caracteristicilor
feature_names = x_train.columns

# Creează un obiect de tip DecisionTreeClassifier
clf = DecisionTreeClassifier()

# Antrenează clasificatorul
clf.fit(x_train, y_train)



#Definează numele modelelor
models = ['Decision Tree', 'Random Forest']

#Definează acuratețea fiecărui model
accuracies = [metrics.accuracy_score(y_test, y_pred), accuracy_rf]

#Plotează graficul
plt.bar(models, accuracies, color=['blue', 'green'])

#Adaugă etichete și titlu
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')

#Afișează graficul
plt.show()

###############################################################################

# Selectează caracteristicile pentru regresie
features2 = df_encoded.drop(['gameid', 'date', 'teamid', 'team', 'home', 'away', '%FGA 2PT'], axis=1)

# Selectează variabila țintă
target2 = df_encoded['%FGA 2PT']

# Împarte datele în seturi de antrenare și de testare
x_train2, x_test2, y_train2, y_test2 = train_test_split(features2, target2, test_size=0.2, random_state=0)
###############################################################################

#Regresie Liniara
model = LinearRegression()

model.fit(x_train2, y_train2)

y_prezis2 = model.predict(x_test2)

r_patrat2 = model.score(x_train2, y_train2)    

scorul_r2_pts2 = r2_score(y_test2, y_prezis2)

print(f'Coeficientul de determinare (R^2): {scorul_r2_pts2}')

#Valorile prezise vs. cele reale
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test2, y=y_prezis2)
plt.xlabel('Puncte Reale')
plt.ylabel('Puncte Prezise')
plt.title('Puncte Reale vs. Puncte Prezise')
plt.show()
###############################################################################

#Regresie OLS
#Adauga o coloană constanta pentru termenul liber (intercept)
x_train2 = sm.add_constant(x_train2)
x_test2 = sm.add_constant(x_test2)

#Initializeaza si antreneaza modelul de regresie liniara folosind OLS
ols_model2 = sm.OLS(y_train2, x_train2).fit()

#Realizeaza predictii pe setul de testare
y_pred2_pts2 = ols_model2.predict(x_test2)

#Evaluează performanta modelului
r2rOLS_pts2 = r2_score(y_test2, y_pred2_pts2)

print(ols_model2.summary())
print(f'Coeficientul de determinare (R^2): {r2rOLS_pts2}')

# Vizualizează valorile prezise vs. cele reale
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test2, y=y_pred2_pts2)
plt.xlabel('Puncte Reale')
plt.ylabel('Puncte Prezise')
plt.title('Puncte Reale vs. Puncte Prezise (Regresie OLS)')
plt.show()

###############################################################################
#Regresie Random Forest
# Inițializați și antrenați Random Forest Regressor
rf_model_pts2 = RandomForestRegressor(n_estimators=100, random_state=0)
rf_model_pts2.fit(x_train2, y_train2)

# Make predictions on the test set
y_pred_pts2 = rf_model_pts2.predict(x_test2)

#Evaluam performanta folosind metrici precum coeficientul R^2
r2_pts= r2_score(y_test2, y_pred_pts2)

print(f'Coeficientul de determinare R^2: {r2_pts}')

# Creează un grafic scatter pentru regresia Random Forest
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.title('Regresie Random Forest: Valori Reale vs. Prezise')
plt.xlabel('Valori Reale')
plt.ylabel('Valori Prezise')
plt.show()

###############################################################################

#Importanta
feature_importance = rf_model_pts2.feature_importances_

coeficienti = ols_model2.params
p_values = ols_model2.pvalues

rez_regresie = pd.DataFrame({
    "Coeficient": coeficienti,
    "p-values": p_values
    })

r_squared = ols_model2.rsquared

#VIF - pt multicoliniaritate
x_with_const = sm.add_constant(features2)
vif_data = pd.DataFrame()
vif_data["Variable"] = x_with_const.columns
vif_data["VIF"] = [variance_inflation_factor(x_with_const.values, i) for i in range(x_with_const.shape[1])]

print(vif_data)

###############################################################################


#Optimizare set de date dupa rulare VIF
features3 = features2.drop(['%PTS 2PT', '%PTS 2PT MR', '%PTS 3PT', '%PTS FT', '%PTS PITP', '2FGM %AST', '2FGM %UAST', 'FGM %AST','FGM %UAST','type_playoff','type_regular'], axis=1)

#VIF - pt multicoliniaritate
x_with_const = sm.add_constant(features3)
vif_data2 = pd.DataFrame()
vif_data2["Variable"] = x_with_const.columns
vif_data2["VIF"] = [variance_inflation_factor(x_with_const.values, i) for i in range(x_with_const.shape[1])]

print(vif_data2)

#RFE
rfe = RFE(estimator = model, n_features_to_select=10)
rf = rfe.fit(features3, target)

atribute_selectate = pd.DataFrame({
    "atribut": features3.columns,
    "selectate": rfe.support_,
    "ranking": rfe.ranking_
    })

###############################################################################

#Regresie pe modelul optimizat
atribute_selectate_pt_reg = atribute_selectate[atribute_selectate["selectate"]]["atribut"]

features_selected= features3[atribute_selectate_pt_reg]

x_tr_sel, x_test_sel, y_tr_sel, y_test_sel = train_test_split(features_selected, target, test_size = 0.2, random_state=0)

reg_lin_sel = LinearRegression()
reg_lin_sel.fit(x_tr_sel, y_tr_sel)

y_prezis_sel = reg_lin_sel.predict(x_test_sel)
r2_scor_selectat = r2_score(y_test_sel, y_prezis_sel)

print(r2_scor_selectat)

#Valorile prezise vs. cele reale
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test_sel, y=y_prezis_sel)
plt.xlabel('Puncte Reale')
plt.ylabel('Puncte Prezise')
plt.title('Puncte Reale vs. Puncte Prezise Selectate')
plt.show()

###############################################################################

#Regresie OLS selectat
#Adauga o coloană constanta pentru termenul liber (intercept) selectat
x_tr_sel = sm.add_constant(x_tr_sel)
x_test_sel = sm.add_constant(x_test_sel)

#Initializeaza si antreneaza modelul de regresie liniara folosind OLS selectat
ols_model_selectat = sm.OLS(y_tr_sel, x_tr_sel).fit()

#Realizeaza predictii pe setul de testare selectat
y_pred2_selectat = ols_model_selectat.predict(x_test_sel)

#Evaluează performanta modelului selectat
r2rOLS_selectat = r2_score(y_test_sel, y_pred2_selectat)

print(ols_model_selectat.summary())
print(f'Coeficientul de determinare (R^2): {r2rOLS_selectat}')

# Vizualizează valorile prezise vs. cele reale
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test_sel, y=y_pred2_selectat)
plt.xlabel('Puncte Reale')
plt.ylabel('Puncte Prezise')
plt.title('Puncte Reale vs. Puncte Prezise (Regresie OLS) Selectate')
plt.show()

###############################################################################

#Regresie Random Forest
# Inițializați și antrenați Random Forest Regressor selectat
rf_model_pts2 = RandomForestRegressor(n_estimators=100, random_state=0)
rf_model_pts2.fit(x_tr_sel, y_tr_sel)

# Make predictions on the test set selected
y_pred_select = rf_model_pts2.predict(x_test_sel)

#Evaluam performanta folosind metrici precum coeficientul R^2 selectat
r2_select = r2_score(y_test_sel, y_pred_select)

print(f'Coeficientul de determinare R^2: {r2_select}')

# Creează un grafic scatter pentru regresia Random Forest
plt.figure(figsize=(10, 6))
plt.scatter(y_test_sel, y_pred_select, alpha=0.5)
plt.title('Regresie Random Forest: Puncte Reale vs. Puncte Selectate')
plt.xlabel('Puncte Reale')
plt.ylabel('Puncte Prezise')
plt.show()
##################################################################################
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Define features and target variable
features = ['%FGA 2PT', '%FGA 3PT', '%PTS 2PT', '%PTS 2PT MR', '%PTS 3PT', 
            '%PTS FBPS', '%PTS FT', '%PTS OFF TO', '%PTS PITP', '2FGM %AST', 
            '2FGM %UAST', '3FGM %AST', '3FGM %UAST', 'FGM %AST', 'FGM %UAST']
target = 'win'




# Split the data into training and testing sets
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


