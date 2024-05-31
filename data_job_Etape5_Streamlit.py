# -*- coding: utf-8 -*-
"""
Created on Sun May 12 22:57:38 2024

@author: Olivia_Anca_Oumou_
"""

import streamlit as st
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, cross_val_score

warnings.filterwarnings("ignore", category=UserWarning)

#fonction d'importation de données avec cache pour plus de rapidité d'exécution
@st.cache_data
def load_data():
  data = pd.read_csv("kaggle_survey_2020_responses.csv", sep =",", low_memory = False)
df = load_data()
 
@st.cache_data
def data_header():
    st.header("Données")

st.sidebar.title("DataJob")
pages=["Présentation du projet", "Données", "Analyse exploratoire et statistique","Transformation des données",
       "Modélisation", "Démo"]
page=st.sidebar.radio("Menu", pages)

st.sidebar.image("Streamlit_DataJob\photo_data.jpg", use_column_width=True)
st.sidebar.text("")


container = st.sidebar.container(border=True)
container.write("Formation continue - Data Analyst - Octobre 2023")
container.write("Olivia SOULABAILLE")
container.write ("Oumou MAGASSA")
container.write("Anca DELVAL-CULCEA")




if page == pages[0]:
    st.title(":orange[DataJob]")
    st.subheader(":orange[Contexte]")
    st.write("En 2020, Kaggle a lancé une enquête majeure sur le Machine Learning et la data science, dans le but de dresser un état des lieux complet du marché de la science des données et de l'apprentissage automatique. Après un processus de nettoyage des données, dont les détails ne sont pas explicités par Kaggle, le jeu de données mis à disposition comprend les réponses de 20 036 participants. Ces participants ont des profils divers : experts ou débutants, en activité professionnelle ou non.")
    st.subheader(":orange[Objectifs du projet]")
    st.write("Ce projet vise principalement à explorer les divers profils techniques présents dans l'industrie de la data en 2020. Nous avons entrepris une analyse approfondie des compétences, des applications pratiques et des outils maîtrisés par chaque poste, dans le but de saisir pleinement la nature de chaque métier du domaine.")
    st.write("Notre objectif était la création d'un outil de recommandation ou de profilage des apprenants en fonction de leurs compétences et de leurs intérêts, tout en nous positionnant nous-mêmes dans l'industrie de la data.")





elif page == pages[1]:
    st.title(":orange[Données]")
    st.subheader (":orange[Jeu de données]")
    url = "https://www.kaggle.com/c/kaggle-survey-2020/overview"
    st.write("Nous avons utilisé le jeu de données disponible sur [kaggle](%s), constitué des réponses des 20 036 participants aux 39 questions de l'enquête. Ces réponses sont réparties sur 355 colonnes. La variable cible sont les métiers situés dans la colonne « Q5 ». " % url)
    df = pd.read_csv("kaggle_survey_2020_responses.csv", sep =",", low_memory = False)
    st.dataframe(df.head())
    st.write("")
    if st.checkbox ("Afficher la description du JDD"):
        st.dataframe(df.describe())
    st.write("")
    st.subheader(":orange[Particularités]")
    st.markdown("""
Notre projet est à 100% un questionnaire, avec des questions à choix multiples ou unique. Nous avons donc dû évaluer la pertinence des réponses et traiter les valeurs manquantes de manière appropriée : 
- Si au moins une réponse avait été saisie dans une question à choix multiple, les autres réponses manquantes pour cette question étaient remplacées par « No »
- Si aucune réponse n'avait pas du tout été saisie pour une question, cette question était considérée comme déclinée et les réponses manquantes étaient remplacées par « Declined Question ».
"""
)

    st.title(":orange[Préparation des données]")
    df0 = pd.read_csv("data_job_donnees_pretraitees_NoNaN0.csv", sep =",", low_memory = False)
    st.subheader(":orange[Nettoyage du JDD]")
    st.markdown("""
Afin d'avoir un jeu de données exploitables, nous avons procédé à quelques suppressions :
- les lignes à valeurs nulles dans la colonne cible (« Q5 »)
- les colonnes vides ou incluant que des « NaN »
- la première ligne qui contenait les questions du questionnaire
"""
)
            
    st.write("")
    
    st.subheader(":orange[Réduction des dimensions]")
    
    st.write("Nous avons exclu des classes non pertinentes pour notre objectif : « Student », « Currently not employed » et « Other ». Cela a conduit à une réduction du nombre de répondants à 10 717, soit une perte d'environ 46 % par rapport au jeu de données initial. ")
    value_counts_q5 = df0['Q5'].value_counts()
    fig1 = px.pie(labels=value_counts_q5.index, values=value_counts_q5.values, names=value_counts_q5.index, color_discrete_sequence=px.colors.qualitative.Pastel)
    fig1.update_layout(title='Répartition des variables cibles avant suppression 3 classes', showlegend=True)
    st.plotly_chart(fig1)
    
    df3 = pd.read_csv("data_job_donnees_pretraitees_NoNaN.csv", sep =",", low_memory = False)
    value_counts_q5 = df3['Q5'].value_counts()
    fig2 = px.pie(labels=value_counts_q5.index, values=value_counts_q5.values, names=value_counts_q5.index, color_discrete_sequence=px.colors.qualitative.Pastel)
    fig2.update_layout(title='Répartition des variables cibles après suppression 3 classes', showlegend=True)
    st.plotly_chart(fig2)
    
    st.write("")
    
    st.markdown("""
Pour assurer une efficacité optimisée du modèle nous avions besoin d'un taux de réponses global maximisé. Nous avons donc enlevé les questions les moins pertinentes par rapport à l'objectif final :
- questions « alternative » (B) à destination des publics non professionnels
- questions additionnelles conditionnées à la réponse d’une question précédente.
"""
)
    df1 = pd.read_csv("data_job_donnees_pretraitees_NoNaN1.csv", sep =",", low_memory = False)
    def calculate_response_percentage(row):
        total_questions = len(row)
        declined_count = row.tolist().count("Declined Question")
        answered_count = total_questions - declined_count
        return (answered_count / total_questions) * 100

    response_percentages = df1.apply(calculate_response_percentage, axis=1)
    
    # boîte à moustaches avec Plotly
    fig3 = px.box(df1, y=response_percentages)
    fig3.update_layout(title="Taux de réponses par répondant après suppression questions alternatives et conditionnées",
                  yaxis_title="Taux de réponses (%)")
    st.plotly_chart(fig3)
    
    st.write("")
    
    st.write("Afin d'optimiser le modèle final, nous avons supprimé les questions dont le taux de réponse était inférieur à 80%.")
    st.info("Suite étape de nettoyage des données : 153 colonnes vs 355 initialement, 28 questions vs 39.")





elif page == pages[2]:
    st.title(":orange[Analyse exploratoire et statistique]")
    st.subheader(":orange[Exploration des données]")
    st.write("Après le nettoyage des données, nous avons une nouvelle répartition des classes :")
    df2 = pd.read_csv("data_job_donnees_pretraitees_NoNaN.csv", sep =",", low_memory = False)
    value_counts_q5 = df2['Q5'].value_counts()
    fig4 = px.pie(labels=value_counts_q5.index, values=value_counts_q5.values, names=value_counts_q5.index, color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig4)
    st.write("Plus généralement, ce jeu de données nous fournit des tendances sur les profils des répondants :")
    st.markdown("""
- La majorité des répondants possèdent un diplôme niveau Master ou Licence (74%)
"""
)
    value_counts_q4 = df2['Q4'].value_counts()
    fig5 = px.bar(x = value_counts_q4.index, y = value_counts_q4.values, 
                  labels = {'x':'Niveau études', 'y' : 'Nombre de répondants'},
                  text_auto=True,
                  color = value_counts_q4.values)
    fig5.update_yaxes (range = [0,4000])
    st.plotly_chart(fig5)
    
    st.write("")
    st.markdown("""
- La majorité des répondants ont une expérience d'écriture de code comprise entre 1 et 10 ans (64%)
"""
)
    value_counts_q6 = df2['Q6'].value_counts()
    fig6 = px.bar(x = value_counts_q6.index, y = value_counts_q6.values, 
                  labels = {'x':'Nombre années écriture code', 'y' : 'Nombre de répondants'},
                  text_auto=True,
                  color = value_counts_q6.values)
    fig6.update_yaxes (range = [0,2500])
    st.plotly_chart(fig6)
    
    st.write("")
    st.markdown("""
- La tranche d'âge la plus représentée est de 22 à 34 ans (55%) 
"""
)    
    value_counts_q1 = df2['Q1'].value_counts()
    fig7 = px.bar(x = value_counts_q1.index, y = value_counts_q1.values, 
                  labels = {'x':'Age', 'y' : 'Nombre de répondants'},
                  text_auto=True,
                  color = value_counts_q1.values)
    fig7.update_yaxes (range = [0,2000])
    st.plotly_chart(fig7)
    
    st.write("")
    st.markdown("""
- La plupart des répondants ont moins de 2 ans d'expérience en apprentissage automatique (58%) 
"""
)    
    value_counts_q15 = df2['Q15'].value_counts()
    fig8 = px.bar(x = value_counts_q15.index, y = value_counts_q15.values, 
                  labels = {"x":"Nombre d'années", "y" : "Nombre de répondants"},
                  text_auto=True,
                  color = value_counts_q15.values)
    fig8.update_yaxes (range = [0,2500])
    st.plotly_chart(fig8)

    st.write("Après avoir examiné le profil global des participants, nous avons cherché à réduire le questionnaire aux dix questions les plus pertinentes.")
    st.write("Pour évaluer la relation entre la variable cible et les variables explicatives, nous avons utilisé le test du Chi2, avec deux méthodes distinctes en fonction du type de question (multiple ou unique):")
    st.image("chi2.png", width=600)
    st.write("")
    if st.checkbox ("<< Cocher cette case pour voir le Top 10 des questions"):
        st.markdown("""
1.	Quel est le niveau le plus élevé d'éducation que vous avez atteint ? 
2.	Sélectionnez les activités qui constituent une partie importante de votre rôle au travail.
3.	Quels langages de programmation utilisez-vous régulièrement ?
4.	Quels environnements de développement intégrés (IDE) utilisez-vous régulièrement ?
5.	Depuis combien d'années utilisez-vous des méthodes d'apprentissage automatique ?
6.	Quel est l'outil principal que vous utilisez pour analyser les données ?
7.	Quels frameworks d'apprentissage automatique utilisez-vous régulièrement ?
8.	Quels algorithmes d'apprentissage automatique utilisez-vous régulièrement ?
9.	Votre employeur intègre-t-il des méthodes d'apprentissage automatique ?
10.	Depuis combien d'années écrivez-vous du code et/ou programmez-vous ?
"""
)





elif page == pages[3]:
    st.title (":orange[Stratégie d'encoding]")
    st.write("")
    st.image("encodage.png", use_column_width=True)
    st.write("")
    st.title (":orange[PCA]")
    st.write("")
    
    st.write ("Nous avons testé le Top10 des questions au travers de PCA :")
    if st.checkbox ("<< Cocher cette case pour revoir le Top 10 des questions"):
        st.markdown("""
1.	Quel est le niveau le plus élevé d'éducation que vous avez atteint ? 
2.	Sélectionnez les activités qui constituent une partie importante de votre rôle au travail.
3.	Quels langages de programmation utilisez-vous régulièrement ?
4.	Quels environnements de développement intégrés (IDE) utilisez-vous régulièrement ?
5.	Depuis combien d'années utilisez-vous des méthodes d'apprentissage automatique ?
6.	Quel est l'outil principal que vous utilisez pour analyser les données ?
7.	Quels frameworks d'apprentissage automatique utilisez-vous régulièrement ?
8.	Quels algorithmes d'apprentissage automatique utilisez-vous régulièrement ?
9.	Votre employeur intègre-t-il des méthodes d'apprentissage automatique ?
10.	Depuis combien d'années écrivez-vous du code et/ou programmez-vous ?
"""
)
    st.image("PCA1.png", use_column_width=True)
    st.image("PCA2.png", use_column_width=True)
    st.write("")
    
    st.warning("Les 6 questions ci-dessous permettent une distinction métier réaliste et cohérente vis-à-vis de l’industrie de la data. Pour autant nous notons une proximité réelle entre certains métiers qui risquent de troubler l’interprétation du modèle.")
    if st.checkbox ("<< Cocher cette case pour voir les 6 questions sélectionnées"):
        st.markdown("""
1.	Quel est le niveau le plus élevé d'éducation que vous avez atteint ? 
2.	Sélectionnez les activités qui constituent une partie importante de votre rôle au travail.
3.	Quels langages de programmation utilisez-vous régulièrement ?
4.	Quels environnements de développement intégrés (IDE) utilisez-vous régulièrement ?
5.	Depuis combien d'années utilisez-vous des méthodes d'apprentissage automatique ?
6.	Quel est l'outil principal que vous utilisez pour analyser les données ?

"""
)
    
   
    
    



elif page == pages[4]:
    st.title(":orange[Modélisation]")
    q5_mapping = {
        'Business Analyst': 0, 'DBA/Database Engineer': 1, 'Data Analyst': 2,
        'Data Engineer': 3, 'Data Scientist': 4, 'Machine Learning Engineer': 5,
        'Product/Project Manager': 6, 'Research Scientist': 7,
        'Software Engineer': 8, 'Statistician': 9
    }

    # Charger les données
    df = pd.read_csv("data_job_donnees_encodéesML.csv", low_memory=False)

    # Mapping inverse pour les affichages
    inverse_mapping = {v: k for k, v in q5_mapping.items()}

    # Compter le nombre de chaque titre
    count_by_title = df['Q5'].map(inverse_mapping).value_counts().reset_index()
    count_by_title.columns = ['Métiers', 'Volume']

    # Créer le graphique
    fig = px.bar(count_by_title, x='Métiers', y='Volume', title='Nombre de répondants par métier')
    fig.update_layout(xaxis=dict(tickangle=45))
    st.plotly_chart(fig)

    st.image("matrice_de_confusion.png", use_column_width=True)
    
    st.write("")
    
    st.write("Réduction du nombre de questions à la suite de l'exploration du Top10 Chi2")

    st.image("comparaison_modeles.png", use_column_width=True)
    
    grouped_columns = {}

    for column in df:
        question = column[:3]
        if question in grouped_columns:
            grouped_columns[question].append(column)
        else:
            grouped_columns[question] = [column]

    # Liste des groupes de questions à garder
    groupes_de_questions_a_garder = ['Q5', 'Q23', 'Q4_', 'Q7_', 'Q9_', 'Q15', 'Q38']
    colonnes_a_garder = []

    for groupe in groupes_de_questions_a_garder:
        if groupe in grouped_columns:
            colonnes_a_garder.extend(grouped_columns[groupe])

    # Créer le nouveau dataframe
    df2 = df[colonnes_a_garder]

    # Définir les données et les cibles
    data2 = df2.drop('Q5', axis=1)
    target2 = df2['Q5']

    # Inverser les valeurs de Q5 en utilisant le mapping inverse
    df2['Q5_inverse'] = df2['Q5'].map(inverse_mapping)

    # Supprimer les métiers ambigus
    df2 = df2[~df2['Q5_inverse'].isin(['Data Engineer', 'Statistician', 'Product/Project Manager'])]

    # Regrouper les métiers similaires
    q5_grouped_mapping = {
        'Business Analyst': 'Analyst',
        'DBA/Database Engineer': 'Engineer',
        'Data Analyst': 'Analyst',
        'Data Scientist': 'Scientist',
        'Machine Learning Engineer': 'Scientist',
        'Research Scientist': 'Scientist',
        'Software Engineer': 'Engineer',
    }

    # Mapper les classes regroupées dans une nouvelle colonne
    df2['Q5_grouped'] = df2['Q5_inverse'].map(q5_grouped_mapping)
    
    st.write("")

    # Compter le nombre de chaque titre
    st.subheader(":orange[Regroupement des métiers similaires]")
    st.markdown("""
- Analyst = Business Analyst / Data Analyst 
- Engineer = DBA/Database Engineer / Software Engineer
- Scientist = Data Scientist / Machine Learning Engineer / Research Scientist
"""
)
    st.write("Suppression des métiers trop ambigus et touche-à-tout : Data Engineer, Statistician, Product/Project Manager")

    count_by_title_df2 = df2['Q5_grouped'].value_counts().reset_index()
    count_by_title_df2.columns = ['Métiers', 'Volume']

    # Créer le graphique
    fig2 = px.bar(count_by_title_df2, x='Métiers', y='Volume', title='Nombre de répondants par métier')
    fig2.update_layout(xaxis=dict(tickangle=45))
    st.plotly_chart(fig2)

    st.title(":orange[Initialisation des modèles]")
    st.write("")
    st.image("accuracy.png", use_column_width=True)

    # Supprimer les colonnes inutiles
    df2.drop(['Q5', 'Q5_inverse'], axis=1, inplace=True)

    # Assurer la cohérence des fonctionnalités pour l'entraînement et les tests
    X = df2.drop('Q5_grouped', axis=1)
    y = df2['Q5_grouped']

    # Séparer les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Trouver les colonnes manquantes dans X_test et les ajouter
    missing_cols = set(X_train.columns) - set(X_test.columns)
    for c in missing_cols:
        X_test[c] = 0

    # Réordonner les colonnes pour qu'elles soient dans le même ordre que dans X_train
    X_test = X_test[X_train.columns]

    # Charger les modèles
    ANN = joblib.load('ANN_model.joblib')
    arbre_decision = joblib.load("Arbre de Décision_model.joblib")
    extra_tree = joblib.load("Extra Trees_model.joblib")
    foret_aleatoire = joblib.load("Forêt Aléatoire_model.joblib")
    gradient_boosting = joblib.load("Gradient Boosting Classifier_model.joblib")
    KNN = joblib.load("K Plus Proches Voisins (KNN)_model.joblib")
    naive_bayes = joblib.load("Naive Bayes_model.joblib")
    regression_logistique = joblib.load("Régression Logistique_model.joblib")
    SVM = joblib.load("SVM_model.joblib")

    # Sélection du modèle
    modele_choisi = st.selectbox(label='Modèle', options=["ANN", "Arbre de décision", "Extra trees", "Forêt Aléatoire", "Gradient Boosting", "K Plus Proches Voisins (KNN)", "Naives Bayes", "Régression Logistique", "SVM"])

    # Fonction pour obtenir les prédictions
    def obtenir_predictions(modele_choisi):
        if modele_choisi == "ANN":
            y_pred = ANN.predict(X_test)
        elif modele_choisi == "Arbre de décision":
            y_pred = arbre_decision.predict(X_test)
        elif modele_choisi == "Extra trees":
            y_pred = extra_tree.predict(X_test)
        elif modele_choisi == "Forêt Aléatoire":
            y_pred = foret_aleatoire.predict(X_test)
        elif modele_choisi == "Gradient Boosting":
            y_pred = gradient_boosting.predict(X_test)
        elif modele_choisi == "K Plus Proches Voisins (KNN)":
            y_pred = KNN.predict(X_test)
        elif modele_choisi == "Naives Bayes":
            y_pred = naive_bayes.predict(X_test)
        elif modele_choisi == "Régression Logistique":
            y_pred = regression_logistique.predict(X_test)
        elif modele_choisi == "SVM":
            y_pred = SVM.predict(X_test)
        return y_pred

    # Obtenir les prédictions en fonction du modèle sélectionné
    y_pred = obtenir_predictions(modele_choisi)

    # Calculer la précision et le rapport de classification
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Afficher les résultats
    st.write("Accuracy", accuracy)
    st.write("Rapport de classification:", class_report)

    # Calculer la matrice de confusion avec toutes les classes possibles
    all_classes = list(set(y_train) | set(y_test))
    conf_matrix = confusion_matrix(y_test, y_pred, labels=all_classes)

    # Convertir la matrice de confusion en dataframe avec les labels appropriés
    conf_matrix_df = pd.DataFrame(conf_matrix, index=all_classes, columns=all_classes)
    conf_matrix_percent = conf_matrix_df.div(conf_matrix_df.sum(axis=1), axis=0) * 100

    # Créer la heatmap
    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(conf_matrix_percent, annot=True, fmt=".2f", cmap=["#e74c3c", "#f1c40f","#2ecc71" ])
    heatmap.set_xlabel('Valeurs Prédites')
    heatmap.set_ylabel('Valeurs Réelles')
    heatmap.set_title('Matrice de Confusion (Pourcentages)')
    st.pyplot(plt)


    # # Calcul des taux à partir de la matrice de confusion
    # TP = conf_matrix[1, 1]
    # FP = conf_matrix[0, 1]
    # TN = conf_matrix[0, 0]
    # FN = conf_matrix[1, 0]

    # TPR = TP / (TP + FN)  # Taux de vrais positifs
    # FPR = FP / (FP + TN)  # Taux de faux positifs
    # TNR = TN / (TN + FP)  # Taux de vrais négatifs
    # FNR = FN / (FN + TP)  # Taux de faux négatifs
    # st.write("Taux de vrais positifs", TPR*100)
    # st.write("Taux de faux positifs", FPR*100)
    # st.write("Taux de vrais négatifs", TNR*100)
    # st.write("Taux de faux négatifs", FNR*100)



elif page == pages[5]:
    
 # Charger les modèles
     ANN = joblib.load('ANN_model.joblib')
     arbre_decision = joblib.load("Arbre de Décision_model.joblib")
     extra_tree = joblib.load("Extra Trees_model.joblib")
     foret_aleatoire = joblib.load("Forêt Aléatoire_model.joblib")
     gradient_boosting = joblib.load("Gradient Boosting Classifier_model.joblib")
     KNN = joblib.load("K Plus Proches Voisins (KNN)_model.joblib")
     naive_bayes = joblib.load("Naive Bayes_model.joblib")
     regression_logistique = joblib.load("Régression Logistique_model.joblib")
     SVM = joblib.load("SVM_model.joblib")
    
     st.subheader("Veuillez sélectionner un modèle et répondre aux questions pour effectuer une prédiction.") 

     modele_choisi = st.selectbox(label='Modèle', options=["ANN", "Arbre de décision", "Extra trees", "Forêt Aléatoire", "Gradient Boosting", "K Plus Proches Voisins (KNN)", "Naives Bayes", "Régression Logistique", "SVM"])


     if modele_choisi == "ANN":
          modele_selectionne = ANN
     elif modele_choisi == "Arbre de décision":
          modele_selectionne = arbre_decision
     elif modele_choisi == "Extra trees":
          modele_selectionne =  extra_tree
     elif modele_choisi == "Forêt Aléatoire":
          modele_selectionne = foret_aleatoire
     elif modele_choisi == "Gradient Boosting":
          modele_selectionne = gradient_boosting
     elif modele_choisi == "K Plus Proches Voisins (KNN)":
          modele_selectionne =  KNN
     elif modele_choisi == "Naives Bayes":
          modele_selectionne =  naive_bayes
     elif modele_choisi == "Régression Logistique":
          modele_selectionne =  regression_logistique
     elif modele_choisi == "SVM":
          modele_selectionne =  SVM
 
     mapping_multiple = {0: 'Declined', 1: 'No', 2: 'Yes'}
     mapping_unique = {0: 'No', 1: 'Yes'}

# Mapping inverse pour convertir les réponses en valeurs numériques pour les modèles
     inverse_mapping_multiple = {v: k for k, v in mapping_multiple.items()}
     inverse_mapping_unique = {v: k for k, v in mapping_unique.items()}

# Pour rappel : encoding
# question choix multiple Q23, Q4, Q7 & Q9 --> valeur 0 = Declined / 1 = No / 2 = Yes
# question choix unique Q4, Q15 & Q38 --> valeur 0 = No / 1 = Yes

# Q5','Q23', 'Q4_', 'Q7_','Q9_', 'Q15', 'Q38']

# Définir les questions du questionnaire et leurs options
     mapping_questions = {
         '1 - Select any activities that make up an important part of your role at work: (Select all that apply)': {
             'Analyze and understand data to influence product or business decisions': 'Q23_Analyze and understand data to influence product or business decisions',
             'Build and/or run the data infrastructure that my business uses for storing, analyzing, and operationalizing data': 'Q23_Build and/or run the data infrastructure that my business uses for storing, analyzing, and operationalizing data',
             'Build prototypes to explore applying machine learning to new areas': 'Q23_Build prototypes to explore applying machine learning to new areas',
             'Build and/or run a machine learning service that operationally improves my product or workflows': 'Q23_Build and/or run a machine learning service that operationally improves my product or workflows',
             'Experimentation and iteration to improve existing ML models': 'Q23_Experimentation and iteration to improve existing ML models',
             'Do research that advances the state of the art of machine learning': 'Q23_Do research that advances the state of the art of machine learning',
             'None of these activities are an important part of my role at work': 'Q23_None of these activities are an important part of my role at work',
             'Other': 'Q23_Other'},
         
         '2 - What is the highest level of formal education that you have attained or plan to attain within the next 2 years?': {
             'Bachelor’s degree': 'Q4_Bachelor’s degree',
             'Doctoral degree': 'Q4_Doctoral degree',
             'I prefer not to answer': 'Q4_I prefer not to answer',
             'Master’s degree': 'Q4_Master’s degree',
             'No formal education past high school': 'Q4_No formal education past high school',
             'Professional degree': 'Q4_Professional degree',
             'Some college/university study without earning a bachelor’s degree': 'Q4_Some college/university study without earning a bachelor’s degree',},
         
         '3 - What programming languages do you use on a regular basis? (Select all that apply)': {
             'Python': 'Q7_Python',
             'R': 'Q7_R',
             'SQL': 'Q7_SQL',
             'C': 'Q7_C',
             'C++': 'Q7_C++',
             'Java': 'Q7_Java',
             'Javascript': 'Q7_Javascript',
             'Julia': 'Q7_Julia',
             'Swift': 'Q7_Swift',
             'Bash': 'Q7_Bash',
             'MATLAB': 'Q7_MATLAB',
             'Other': 'Q7_Other',},
         
         "4 - Which of the following integrated development environments (IDE's) do you use on a regular basis? (Select all that apply)": {
             'Jupyter (JupyterLab, Jupyter Notebooks, etc)': 'Q9_Jupyter (JupyterLab, Jupyter Notebooks, etc) ',
             'RStudio': 'Q9_RStudio ',
             'Visual Studio': 'Q9_Visual Studio',
             'Visual Studio Code (VSCode)': 'Q9_Visual Studio Code (VSCode)',
             'PyCharm': 'Q9_PyCharm ',
             'Spyder':  'Q9_Spyder  ',
             'Notepad++': 'Q9_Notepad++  ',
             'Sublime Text': 'Q9_Sublime Text  ',
             'Vim / Emacs': 'Q9_Vim / Emacs  ',
             'MATLAB': 'Q9_MATLAB ',
             'Other': 'Q9_Other',},
         
         "5 - For how many years have you used machine learning methods?": {
             '1-2 years': 'Q15_1-2 years',
             '10-20 years': 'Q15_10-20 years',
             '2-3 years': 'Q15_2-3 years',
             '20 or more years': 'Q15_20 or more years',
             '3-4 years': 'Q15_3-4 years',
             '4-5 years': 'Q15_4-5 years',
             '5-10 years': 'Q15_5-10 years',
             'I do not use machine learning methods': 'Q15_I do not use machine learning methods',
             'Under 1 year': 'Q15_Under 1 year',},
         
         "6 - What is the primary tool that you use at work or school to analyze data?": {
             'Advanced statistical software (SPSS, SAS, etc.)': 'Q38_Advanced statistical software (SPSS, SAS, etc.)',
             'Basic statistical software (Microsoft Excel, Google Sheets, etc.)': 'Q38_Basic statistical software (Microsoft Excel, Google Sheets, etc.)',
             'Business intelligence software (Salesforce, Tableau, Spotfire, etc.)': 'Q38_Business intelligence software (Salesforce, Tableau, Spotfire, etc.)',
             'Cloud-based data software & APIs (AWS, GCP, Azure, etc.)': 'Q38_Cloud-based data software & APIs (AWS, GCP, Azure, etc.)',
             'Local development environments (RStudio, JupyterLab, etc.)': 'Q38_Local development environments (RStudio, JupyterLab, etc.)',
             'Other': 'Q38_Other'}
     }


# Créer un formulaire pour les réponses de l'utilisateur

     responses = {}

     for section, qs in mapping_questions.items():
         st.header(section)
         for display_text, original_key in qs.items():
             if 'Q7' in original_key or 'Q9' in original_key or 'Q23' in original_key: 
                 response = st.radio(display_text, list(mapping_multiple.values()),key=original_key)
                 responses[original_key] = inverse_mapping_multiple[response]
             else: 
                 response = st.radio(display_text, list(mapping_unique.values()),key=original_key)
                 responses[original_key] = inverse_mapping_unique[response]
        
     if st.button('Prédire'):
        new_row = pd.DataFrame([responses])
        if modele_selectionne is not None:
            prediction = modele_selectionne.predict(new_row)
            probabilites = modele_selectionne.predict_proba(new_row)
            classes = modele_selectionne.classes_

    
            st.subheader(f'Le modèle {modele_choisi} prédit : {prediction}')
            st.subheader(f'Les probalités du modèle {modele_choisi} sont : ')
            for classe, probabilite in zip(classes, probabilites[0]):
                st.subheader(f'{classe} : {probabilite* 100:.2f}%')

    
    

    
    
    
    
    
    
