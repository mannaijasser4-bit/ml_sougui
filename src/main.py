import pyodbc
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
import os
import warnings
warnings.filterwarnings('ignore')

# =========================
# CRÉATION DES DOSSIERS
# =========================
os.makedirs('reports', exist_ok=True)
os.makedirs('models', exist_ok=True)


# =========================
# CONNEXION BASE DE DONNÉES
# =========================
def connect_to_sql_server(server, database, username, password):
    """
    Connexion à SQL Server
    """
    conn_str = f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}'
    return pyodbc.connect(conn_str)


# =========================
# CHARGEMENT DES DONNÉES
# =========================
def load_data(conn):
    """
    Chargement des données B2C et B2B depuis SQL Server
    """

    query_b2c = "SELECT CustomerID, PurchaseDate, Amount FROM dbo.Fact_V_B2C"
    query_b2b = "SELECT CustomerID, PurchaseDate, Amount FROM dbo.Fact_V_B2B"

    df_b2c = pd.read_sql(query_b2c, conn)
    df_b2b = pd.read_sql(query_b2b, conn)

    # Fusion des données (B2C + B2B)
    df = pd.concat([df_b2c, df_b2b], ignore_index=True)

    # =========================
    # DATA CLEANING (NETTOYAGE)
    # =========================

    # Conversion des dates
    df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'])

    # Suppression des valeurs manquantes
    df = df.dropna()

    # Suppression des doublons
    df = df.drop_duplicates()

    return df


# =========================
# FEATURE ENGINEERING (RFM)
# =========================
def calculate_rfm(df):
    """
    Création des variables RFM :

    R = Récence (temps depuis dernier achat)
    F = Fréquence (nombre d'achats)
    M = Montant total dépensé
    """

    max_date = df['PurchaseDate'].max()

    rfm = df.groupby('CustomerID').agg(
        Recency=('PurchaseDate', lambda x: (max_date - x.max()).days),
        Frequency=('PurchaseDate', 'count'),
        Monetary=('Amount', 'sum')
    )

    rfm['Recency'] = rfm['Recency'].astype(int)

    return rfm


# =========================
# MISE À L'ÉCHELLE DES DONNÉES
# =========================
def scale_features(df):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    return scaled_data, scaler


# =========================
# CLUSTERING (NON SUPERVISÉ)
# =========================
def perform_clustering(rfm_scaled):
    """
    KMeans + Clustering Hiérarchique
    """

    # Méthode du coude (Elbow Method)
    wcss = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(rfm_scaled)
        wcss.append(kmeans.inertia_)

    plt.figure()
    plt.plot(range(1, 11), wcss, marker='o')
    plt.title("Méthode du coude (Elbow Method)")
    plt.xlabel("Nombre de clusters")
    plt.ylabel("WCSS")
    plt.savefig('reports/elbow_method.png')
    plt.close()

    # Choix du nombre de clusters
    optimal_k = 4
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    kmeans_labels = kmeans.fit_predict(rfm_scaled)

    # Score silhouette (qualité clustering)
    sil_score = silhouette_score(rfm_scaled, kmeans_labels)
    print("Score Silhouette KMeans :", sil_score)

    # Clustering hiérarchique
    linkage_matrix = linkage(rfm_scaled, method='ward')

    plt.figure()
    dendrogram(linkage_matrix)
    plt.title("Dendrogramme (Clustering Hiérarchique)")
    plt.savefig('reports/dendrogram.png')
    plt.close()

    hierarchical = AgglomerativeClustering(n_clusters=optimal_k)
    hierarchical_labels = hierarchical.fit_predict(rfm_scaled)

    return kmeans_labels, hierarchical_labels, optimal_k


# =========================
# SEGMENTATION CLIENTS (BUSINESS)
# =========================
def label_customers(rfm, kmeans_labels):
    """
    Transformation des clusters en segments business
    """

    rfm_cluster = rfm.copy()
    rfm_cluster['cluster'] = kmeans_labels

    summary = rfm_cluster.groupby('cluster').agg(
        Recency=('Recency', 'mean'),
        Frequency=('Frequency', 'mean'),
        Monetary=('Monetary', 'mean')
    )

    # Score business (valeur client)
    summary['score'] = summary['Monetary'] - 0.1 * summary['Recency']

    top_cluster = summary['score'].idxmax()

    # Attribution des labels
    labels = [
        "Top Value" if c == top_cluster else "Standard"
        for c in kmeans_labels
    ]

    return pd.Series(labels, index=rfm.index)


# =========================
# MACHINE LEARNING SUPERVISÉ
# =========================
def perform_classification(rfm_scaled, customer_labels):

    # Encodage des labels
    label_mapping = {'Top Value': 3}
    y = customer_labels.map(label_mapping).fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        rfm_scaled, y, test_size=0.3, random_state=42
    )

    # =========================
    # RANDOM FOREST
    # =========================
    rf_params = {'n_estimators': [100, 200], 'max_depth': [10, 20, None]}
    rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5)
    rf_grid.fit(X_train, y_train)
    rf_best = rf_grid.best_estimator_

    # =========================
    # REGRESSION LOGISTIQUE
    # =========================
    lr_params = {'C': [0.1, 1, 10]}
    lr_grid = GridSearchCV(LogisticRegression(), lr_params, cv=5)
    lr_grid.fit(X_train, y_train)
    lr_best = lr_grid.best_estimator_

    # =========================
    # ÉVALUATION
    # =========================
    rf_pred = rf_best.predict(X_test)
    lr_pred = lr_best.predict(X_test)

    # Matrice de confusion
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(confusion_matrix(y_test, rf_pred), annot=True, ax=ax[0])
    ax[0].set_title("Random Forest")

    sns.heatmap(confusion_matrix(y_test, lr_pred), annot=True, ax=ax[1])
    ax[1].set_title("Régression Logistique")

    plt.savefig("reports/confusion_matrices.png")
    plt.close()

    # ROC-AUC (multiclasse)
    rf_auc = roc_auc_score(y_test, rf_best.predict_proba(X_test), multi_class='ovr')
    lr_auc = roc_auc_score(y_test, lr_best.predict_proba(X_test), multi_class='ovr')

    print("ROC-AUC Random Forest :", rf_auc)
    print("ROC-AUC Logistic Regression :", lr_auc)

    return rf_best, lr_best


# =========================
# PIPELINE PRINCIPAL
# =========================
if __name__ == "__main__":

    # Connexion base de données
    try:
        conn = connect_to_sql_server(
            'AKTEKSB\\ASUS',
            'Sougui_DWH',
            'sa',
            'your_password'
        )

        df = load_data(conn)
        conn.close()

    except:
        print("Utilisation des données simulées...")

        np.random.seed(42)
        df = pd.DataFrame({
            'CustomerID': range(1, 101),
            'PurchaseDate': pd.date_range('2023-01-01', periods=100),
            'Amount': np.random.uniform(10, 1000, 100)
        })

    # =========================
    # FEATURE ENGINEERING
    # =========================
    rfm = calculate_rfm(df)

    # Normalisation
    rfm_scaled, scaler = scale_features(rfm)

    # =========================
    # CLUSTERING
    # =========================
    kmeans_labels, hierarchical_labels, optimal_k = perform_clustering(rfm_scaled)

    # Segmentation business
    customer_labels = label_customers(rfm, kmeans_labels)

    # =========================
    # CLASSIFICATION
    # =========================
    rf_model, lr_model = perform_classification(rfm_scaled, customer_labels)

    # =========================
    # SAUVEGARDE DES MODÈLES
    # =========================
    import joblib
    joblib.dump(rf_model, 'models/random_forest.pkl')
    joblib.dump(lr_model, 'models/logistic_regression.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')