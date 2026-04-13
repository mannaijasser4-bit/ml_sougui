import pyodbc
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
import os
import warnings
warnings.filterwarnings('ignore')

# Ensure directories exist
os.makedirs('reports', exist_ok=True)
os.makedirs('models', exist_ok=True)

def connect_to_sql_server(server, database, username, password):
    """
    Connect to SQL Server using pyodbc.
    """
    conn_str = f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}'
    conn = pyodbc.connect(conn_str)
    return conn

def load_data(conn):
    """
    Load B2C and B2B data from SQL Server.
    Assume tables have columns: CustomerID, PurchaseDate, Amount
    """
    query_b2c = "SELECT CustomerID, PurchaseDate, Amount FROM dbo.Fact_V_B2C"
    query_b2b = "SELECT CustomerID, PurchaseDate, Amount FROM dbo.Fact_V_B2B"

    df_b2c = pd.read_sql(query_b2c, conn)
    df_b2b = pd.read_sql(query_b2b, conn)

    # Merge/Union the data
    df = pd.concat([df_b2c, df_b2b], ignore_index=True)
    df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'])
    return df

def calculate_rfm(df):
    r"""
    Calculate RFM metrics.

    Recency (R): Days since last purchase.
    Frequency (F): Number of purchases.
    Monetary (M): Total amount spent.

    Formulas:
    - $ R = \max(\text{PurchaseDate}) - \text{last_purchase_date} $
    - $ F = \text{count(purchases per customer)} $
    - $ M = \sum(\text{Amount per customer}) $
    """
    max_date = df['PurchaseDate'].max()
    rfm = df.groupby('CustomerID').agg(
        Recency=('PurchaseDate', lambda x: (max_date - x.max()).days),
        Frequency=('PurchaseDate', 'count'),
        Monetary=('Amount', 'sum'),
    )
    rfm['Recency'] = rfm['Recency'].astype(int)
    return rfm


def scale_features(df):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    return scaled_data, scaler

def perform_clustering(rfm_scaled):
    """
    Perform K-Means and Hierarchical Clustering.
    Evaluate with Elbow Method and Silhouette Score.
    """
    # Elbow Method for K-Means
    wcss = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(rfm_scaled)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 11), wcss, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.savefig('reports/elbow_method.png')
    plt.close()

    # Optimal k from Elbow, say 4
    optimal_k = 4
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    kmeans_labels = kmeans.fit_predict(rfm_scaled)

    # Silhouette Score
    sil_score = silhouette_score(rfm_scaled, kmeans_labels)
    print(f'Silhouette Score for K-Means: {sil_score}')

    # Hierarchical Clustering
    linkage_matrix = linkage(rfm_scaled, method='ward')
    plt.figure(figsize=(10, 7))
    dendrogram(linkage_matrix)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.savefig('reports/dendrogram.png')
    plt.close()

    hierarchical = AgglomerativeClustering(n_clusters=optimal_k)
    hierarchical_labels = hierarchical.fit_predict(rfm_scaled)

    # Scatter plot for K-Means
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=rfm_scaled.iloc[:, 0], y=rfm_scaled.iloc[:, 1], hue=kmeans_labels, palette='viridis')
    plt.title('K-Means Clustering Scatter Plot')
    plt.savefig('reports/kmeans_scatter.png')
    plt.close()

    return kmeans_labels, hierarchical_labels, optimal_k

def label_customers(rfm, kmeans_labels):
    """
    Label customers based on the strongest cluster.
    """
    rfm_with_cluster = rfm.copy()
    rfm_with_cluster['cluster'] = kmeans_labels
    summary = rfm_with_cluster.groupby('cluster').agg(
        Recency=('Recency', 'mean'),
        Frequency=('Frequency', 'mean'),
        Monetary=('Monetary', 'mean'),
    )
    summary['score'] = summary['Monetary'] - 0.1 * summary['Recency']
    top_cluster = summary['score'].idxmax()
    return pd.Series(
        ['Top Value' if c == top_cluster else 'Standard' for c in kmeans_labels],
        index=rfm.index,
    )


def perform_classification(rfm_scaled, customer_labels):
    """
    Perform classification using Random Forest and Logistic Regression.
    Use GridSearchCV for hyperparameter tuning.
    """
    # Encode labels
    label_mapping = {'Low Value': 0, 'Medium Value': 1, 'High Value': 2, 'Top Value': 3}
    y = customer_labels.map(label_mapping)

    X_train, X_test, y_train, y_test = train_test_split(rfm_scaled, y, test_size=0.3, random_state=42)

    # Random Forest
    rf_params = {'n_estimators': [100, 200], 'max_depth': [10, 20, None]}
    rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5)
    rf_grid.fit(X_train, y_train)
    rf_best = rf_grid.best_estimator_

    # Logistic Regression
    lr_params = {'C': [0.1, 1, 10]}
    lr_grid = GridSearchCV(LogisticRegression(random_state=42), lr_params, cv=5)
    lr_grid.fit(X_train, y_train)
    lr_best = lr_grid.best_estimator_

    # Predictions
    rf_pred = rf_best.predict(X_test)
    lr_pred = lr_best.predict(X_test)

    # Confusion Matrix
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(confusion_matrix(y_test, rf_pred), annot=True, ax=ax[0])
    ax[0].set_title('Random Forest Confusion Matrix')
    sns.heatmap(confusion_matrix(y_test, lr_pred), annot=True, ax=ax[1])
    ax[1].set_title('Logistic Regression Confusion Matrix')
    plt.savefig('reports/confusion_matrices.png')
    plt.close()

    # ROC-AUC (for multiclass, use macro)
    rf_proba = rf_best.predict_proba(X_test)
    lr_proba = lr_best.predict_proba(X_test)
    rf_auc = roc_auc_score(y_test, rf_proba, multi_class='ovr')
    lr_auc = roc_auc_score(y_test, lr_proba, multi_class='ovr')
    print(f'Random Forest ROC-AUC: {rf_auc}')
    print(f'Logistic Regression ROC-AUC: {lr_auc}')

    # Feature Importance for Random Forest
    importances = rf_best.feature_importances_
    plt.figure(figsize=(8, 6))
    plt.barh(rfm_scaled.columns, importances)
    plt.title('Feature Importance - Random Forest')
    plt.savefig('reports/feature_importance.png')
    plt.close()

    return rf_best, lr_best

if __name__ == "__main__":
    # Connection details
    server = r'AKTEKSB\\ASUS'
    database = 'Sougui_DWH'
    username = 'sa'
    password = 'your_password'  # Replace with actual password

    try:
        conn = connect_to_sql_server(server, database, username, password)
        df = load_data(conn)
        conn.close()
    except Exception:
        print("Using sample data for demo.")
        np.random.seed(42)
        df = pd.DataFrame({
            'CustomerID': range(1, 101),
            'PurchaseDate': pd.date_range('2023-01-01', periods=100, freq='D'),
            'Amount': np.random.uniform(10, 1000, 100)
        })

    rfm = calculate_rfm(df)
    rfm_scaled, scaler = scale_features(rfm)

    kmeans_labels, hierarchical_labels, optimal_k = perform_clustering(rfm_scaled)

    customer_labels = label_customers(rfm, kmeans_labels)

    rf_model, lr_model = perform_classification(rfm_scaled, customer_labels)

    # Save models
    import joblib
    joblib.dump(rf_model, 'models/random_forest.pkl')
    joblib.dump(lr_model, 'models/logistic_regression.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')