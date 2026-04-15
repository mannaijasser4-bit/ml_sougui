import os
import warnings
import streamlit as st
import pyodbc
import pandas as pd

try:
    import numpy as np
except ImportError as exc:
    raise ImportError(
        "Numpy import failed. This often happens because of incompatible Windows DLLs. "
        "Please use a clean virtual environment or conda environment with a compatible Python version. "
        "Example: conda create -n sougui python=3.11 numpy pandas scikit-learn streamlit statsmodels prophet"
    ) from exc

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, confusion_matrix, roc_auc_score, roc_curve, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib
from scipy.cluster.hierarchy import dendrogram, linkage
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# Custom CSS for light theme
st.markdown("""
<style>
    .stApp {
        background-color: #ffffff;
        color: #000000;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stSidebar {
        background-color: #f0f0f0;
        color: #000000;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    .stContainer {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stMetric {
        background-color: #e9ecef;
        border-radius: 8px;
        padding: 10px;
    }
    .stDataFrame {
        background-color: #e9ecef;
        border-radius: 8px;
    }
    .stPlotlyChart {
        background-color: #e9ecef;
        border-radius: 8px;
    }
    h1, h2, h3 {
        color: #007bff;
    }
    .sidebar .sidebar-content {
        background-color: #f0f0f0;
    }
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title='Tableau de Bord Sougui', page_icon='📊', layout='wide')

MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
RF_PATH = os.path.join(MODEL_DIR, 'random_forest.pkl')
XGB_PATH = os.path.join(MODEL_DIR, 'xgboost.pkl')

SERVER = r'localhost'
DATABASE = 'Sougui_DWH'
USERNAME = 'sa'

# Sidebar
st.sidebar.title('📊 Sougui Dashboard')
password = st.sidebar.text_input('Mot de Passe SQL', type='password')
refresh_button = st.sidebar.button('🔄 Actualiser Données')

# Connection status
conn_status = None
if password:
    try:
        conn_str = (
            rf'DRIVER={{SQL Server}};'
            f'SERVER={SERVER};'
            f'DATABASE={DATABASE};'
            f'UID={USERNAME};'
            f'PWD={password}'
        )
        conn = pyodbc.connect(conn_str)
        conn.close()
        conn_status = True
        st.sidebar.success('🟢 Connecté à SQL Server')
    except Exception:
        conn_status = False
        st.sidebar.error('🔴 Connexion échouée')
else:
    st.sidebar.warning('Entrez le mot de passe SQL')

st.sidebar.markdown('---')
st.sidebar.latex(r'''
\text{Métriques RFM:}
\\
R = \max(\text{Date}) - \text{dernier\_achat}
\\
F = \text{count}(\text{achats})
\\
M = \sum(\text{Montant})
''')

pages = ['📊 Résumé Exécutif', '🎯 Intelligence Client', '🤖 Moteur Prédictif', '📉 Prévision Marché']
selected_page = st.sidebar.radio('Navigation', pages)

st.sidebar.markdown('---')
st.sidebar.write('Utilisez les onglets pour explorer les insights.')

def connect_to_sql(password: str):
    if not password:
        return None
    try:
        conn_str = (
            rf'DRIVER={{SQL Server}};'
            f'SERVER={SERVER};'
            f'DATABASE={DATABASE};'
            f'UID={USERNAME};'
            f'PWD={password}'
        )
        return pyodbc.connect(conn_str)
    except Exception as exc:
        st.error(f'Connexion échouée: {exc}')
        return None

@st.cache_data
def load_raw_data(password: str):
    conn = connect_to_sql(password)
    if conn is None:
        return pd.DataFrame()

    query = (
        'SELECT Client_Key AS CustomerID, Date_Key AS PurchaseDate, Montant_total_de_la_commande AS Amount '
        'FROM dbo.Fact_V_B2C '
        'UNION ALL '
        'SELECT Id_Entreprise AS CustomerID, Date_Key AS PurchaseDate, Total_TTC AS Amount '
        'FROM dbo.Fact_V_B2B'
    )
    try:
        df = pd.read_sql(query, conn)
        conn.close()
        df = df[['CustomerID', 'PurchaseDate', 'Amount']].copy()
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')

        # Parse PurchaseDate robustly and avoid Unix epoch fallback when possible
        raw_dates = df['PurchaseDate']
        if pd.api.types.is_numeric_dtype(raw_dates):
            parsed = pd.to_datetime(raw_dates, unit='s', errors='coerce')
            if parsed.isna().all():
                parsed = pd.to_datetime(raw_dates.astype(str), format='%Y%m%d', errors='coerce')
            df['PurchaseDate'] = parsed
        else:
            df['PurchaseDate'] = pd.to_datetime(raw_dates, errors='coerce')
            if df['PurchaseDate'].isna().any():
                fallback = pd.to_datetime(raw_dates.astype(str), format='%Y%m%d', errors='coerce')
                df.loc[df['PurchaseDate'].isna(), 'PurchaseDate'] = fallback

        df = df.dropna(subset=['CustomerID', 'PurchaseDate', 'Amount'])
        return df
    except Exception as exc:
        st.error(f'Erreur lors du chargement des données: {exc}')
        return pd.DataFrame()

def create_sample_data():
    np.random.seed(42)
    df = pd.DataFrame({
        'CustomerID': np.random.choice(range(1, 101), 300),
        'PurchaseDate': pd.date_range(start='2024-01-01', periods=300, freq='D'),
        'Amount': np.random.uniform(50, 3000, 300),
    })
    df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'])
    return df

@st.cache_data
def calculate_rfm(df: pd.DataFrame):
    if df.empty:
        return pd.DataFrame()
    max_date = df['PurchaseDate'].max()
    rfm = df.groupby('CustomerID').agg(
        Recency=('PurchaseDate', lambda x: (max_date - x.max()).days),
        Frequency=('PurchaseDate', 'count'),
        Monetary=('Amount', 'sum'),
    ).reset_index()
    rfm['Recency'] = rfm['Recency'].astype(int)
    return rfm

def scale_features(rfm: pd.DataFrame):
    if rfm.empty:
        return pd.DataFrame(), None
    scaler = StandardScaler()
    scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
    scaled_df = pd.DataFrame(scaled, columns=['Recency', 'Frequency', 'Monetary'])
    return scaled_df, scaler

def adf_test(series: pd.Series):
    try:
        from statsmodels.tsa.stattools import adfuller
    except Exception as exc:
        return None, str(exc)
    try:
        result = adfuller(series.dropna())
        return result[0], result[1]
    except ValueError as ve:
        return None, str(ve)


def select_diff_order(series: pd.Series):
    stat, p_value = adf_test(series)
    if p_value is None or not isinstance(p_value, (int, float, np.floating)):
        return 0, stat, p_value
    return (0 if p_value <= 0.05 else 1), stat, p_value


def ensure_datetime_index(df: pd.DataFrame, date_col: str):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    if df[date_col].isna().any():
        df[date_col] = pd.to_datetime(df[date_col].astype(str), format='%Y%m%d', errors='coerce')
    df = df.dropna(subset=[date_col])
    df = df.sort_values(date_col)
    df = df.set_index(date_col)
    return df


def cluster_profile(rfm: pd.DataFrame, labels: np.ndarray):
    profile = rfm.assign(cluster=labels).groupby('cluster').agg(
        Count=('CustomerID', 'count'),
        AvgRecency=('Recency', 'mean'),
        AvgFrequency=('Frequency', 'mean'),
        AvgMonetary=('Monetary', 'mean'),
    ).reset_index()
    profile['Segment'] = profile.apply(
        lambda row: 'Premium' if row['AvgMonetary'] > profile['AvgMonetary'].quantile(0.75) else (
            'Value' if row['AvgMonetary'] > profile['AvgMonetary'].quantile(0.5) else 'Standard'
        ),
        axis=1,
    )
    return profile

def run_clustering(rfm_scaled: pd.DataFrame, n_clusters: int = 4):
    if rfm_scaled.empty:
        return [], 0.0, np.array([]), None
    wcss = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(rfm_scaled)
        wcss.append(kmeans.inertia_)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(rfm_scaled)
    silhouette = silhouette_score(rfm_scaled, labels)
    
    # Hierarchical
    linkage_matrix = linkage(rfm_scaled, method='ward')
    return wcss, silhouette, labels, linkage_matrix

def plot_dendrogram(linkage_matrix):
    fig, ax = plt.subplots(figsize=(10, 6))
    dendrogram(linkage_matrix, ax=ax)
    ax.set_title('Dendrogramme - Clustering Hiérarchique')
    ax.set_xlabel('Clients')
    ax.set_ylabel('Distance')
    return fig

def get_top_cluster(rfm: pd.DataFrame, labels: np.ndarray):
    if rfm.empty:
        return np.array([])
    summary = rfm.assign(cluster=labels).groupby('cluster').agg(
        Monetary=('Monetary', 'mean'),
        Recency=('Recency', 'mean'),
        Frequency=('Frequency', 'mean'),
    )
    return int(summary['Monetary'].idxmax())

def train_models(rfm_scaled: pd.DataFrame, target: pd.Series):
    if rfm_scaled.empty or target.nunique() < 2:
        return None, None, None, None, None, None, None, None
    X_train, X_test, y_train, y_test = train_test_split(
        rfm_scaled, target, test_size=0.2, random_state=42, stratify=target
    )
    rf_params = {'n_estimators': [100, 200], 'max_depth': [10, 20, None]}
    xgb_params = {'learning_rate': [0.1, 0.2], 'n_estimators': [100, 200], 'subsample': [0.8, 1.0]}
    rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=3)
    xgb_grid = GridSearchCV(XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'), xgb_params, cv=3)
    rf_grid.fit(X_train, y_train)
    xgb_grid.fit(X_train, y_train)
    return rf_grid.best_estimator_, xgb_grid.best_estimator_, X_test, y_test, rf_grid.best_score_, xgb_grid.best_score_, rf_grid, xgb_grid

def get_feature_importances(model, feature_names):
    try:
        importances = np.asarray(model.feature_importances_)
        if importances.shape[0] != len(feature_names):
            importances = importances[:len(feature_names)]
        return importances
    except Exception:
        try:
            booster = model.get_booster()
            score = booster.get_score(importance_type='weight')
            return np.array([score.get(f'f{i}', 0.0) for i in range(len(feature_names))])
        except Exception:
            return np.zeros(len(feature_names))


def plot_confusion_and_roc(model, title: str, X_test, y_test):
    y_pred = model.predict(X_test)
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except Exception:
        y_proba = None
    cm = confusion_matrix(y_test, y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title(f'{title} Matrice de Confusion')
    axes[0].set_xlabel('Prédit')
    axes[0].set_ylabel('Réel')
    if y_proba is not None and len(np.unique(y_test)) > 1:
        auc = roc_auc_score(y_test, y_proba)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        axes[1].plot(fpr, tpr, label=f'AUC = {auc:.2f}')
        axes[1].plot([0, 1], [0, 1], 'k--')
        axes[1].set_title(f'{title} Courbe ROC')
        axes[1].set_xlabel('Taux de Faux Positifs')
        axes[1].set_ylabel('Taux de Vrais Positifs')
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, 'ROC non disponible', horizontalalignment='center', verticalalignment='center')
        axes[1].set_axis_off()
    return fig, y_proba is not None


def aggregate_sales(df: pd.DataFrame):
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'], errors='coerce')
    if df['PurchaseDate'].isna().any():
        df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'].astype(str), format='%Y%m%d', errors='coerce')
    df = df.dropna(subset=['PurchaseDate'])
    ts = df.groupby('PurchaseDate').agg(Sales=('Amount', 'sum')).reset_index()
    ts = ts.sort_values('PurchaseDate')
    ts = ts.set_index('PurchaseDate').asfreq('D').fillna(0)
    return ts

def fit_sarima(series: pd.Series, diff_order: int = 0):
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except Exception as exc:
        return None, str(exc)
    try:
        if len(series.dropna()) < 14:
            return None, 'Série trop courte pour SARIMA.'
        order = (1, diff_order, 0)
        model = SARIMAX(series, order=order, seasonal_order=(0, 0, 0, 0), enforce_stationarity=False, enforce_invertibility=False)
        fit = model.fit(disp=False)
        forecast = fit.get_forecast(steps=30)
        return forecast.predicted_mean, 'SARIMA'
    except Exception as exc:
        return None, str(exc)

def fit_prophet(series: pd.Series):
    try:
        from prophet import Prophet
    except Exception as exc:
        return None, str(exc)
    try:
        if len(series.dropna()) < 14:
            return None, 'Série trop courte pour Prophet.'
        df = series.reset_index()
        df.columns = ['ds', 'y']
        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)
        return forecast['yhat'].tail(30).reset_index(drop=True), 'Prophet'
    except Exception as exc:
        return None, str(exc)

def generate_forecast(series: pd.Series):
    clean = series.dropna()
    if clean.empty:
        return None, 'Aucune donnée de ventes disponible pour la prévision.'
    end_date = clean.index.max()
    forecast_index = pd.date_range(end_date + pd.Timedelta(days=1), periods=30, freq='D')
    if clean.nunique() <= 1:
        const_value = clean.iloc[-1]
        forecast_series = pd.Series([const_value] * 30, index=forecast_index)
        return forecast_series, 'Série constante : prévision basée sur la dernière valeur connue.'
    if clean.shape[0] < 14:
        mean_value = float(clean.mean())
        forecast_series = pd.Series([mean_value] * 30, index=forecast_index)
        return forecast_series, 'Série courte : prévision par moyenne simple.'
    diff_order, adf_stat, adf_p = select_diff_order(clean)
    forecast_series, info = fit_sarima(clean, diff_order=diff_order)
    if forecast_series is not None:
        forecast_series.index = forecast_index
        return forecast_series, f'{info} (ADF p={adf_p:.4f}, d={diff_order})'
    mean_value = float(clean.mean())
    forecast_series = pd.Series([mean_value] * 30, index=forecast_index)
    return forecast_series, f'Prévision moyenne utilisée car SARIMA a échoué : {info}'

def compare_forecasts(series: pd.Series):
    if series.dropna().shape[0] < 20:
        return None, None, 'Série trop courte pour calculer un R² fiable. Veuillez fournir plus de données pour une évaluation précise des modèles.'
    train = series[:-10]
    test = series[-10:]
    diff_order, adf_stat, adf_p = select_diff_order(train)
    sarima_forecast, sarima_info = fit_sarima(train, diff_order=diff_order)
    prophet_forecast, prophet_info = fit_prophet(train)
    results = {}
    if sarima_forecast is not None:
        sarima_forecast = sarima_forecast[:len(test)]
        results['SARIMA'] = {
            'MAE': mean_absolute_error(test, sarima_forecast),
            'RMSE': np.sqrt(mean_squared_error(test, sarima_forecast)),
            'MAPE': mean_absolute_percentage_error(test, sarima_forecast),
            'R2': r2_score(test, sarima_forecast),
            'ADF p-value': adf_p,
            'Diff order': diff_order,
        }
    if prophet_forecast is not None:
        prophet_forecast = prophet_forecast[:len(test)]
        results['Prophet'] = {
            'MAE': mean_absolute_error(test, prophet_forecast),
            'RMSE': np.sqrt(mean_squared_error(test, prophet_forecast)),
            'MAPE': mean_absolute_percentage_error(test, prophet_forecast),
            'R2': r2_score(test, prophet_forecast),
            'ADF p-value': adf_p,
            'Diff order': diff_order,
        }
    return results if results else None, {'SARIMA': sarima_info, 'Prophet': prophet_info}, 'Aucun modèle n\'a pu être ajusté pour la comparaison.'

def is_series_valid(series: pd.Series):
    clean = series.dropna()
    if clean.shape[0] < 10:
        return False, 'Série trop courte pour l\'analyse'
    if clean.nunique() <= 1:
        return False, 'Série constante : pas assez de variation pour ADF/SARIMA'
    return True, None

def recommendation_engine(transactions: pd.DataFrame, customer_id, segment):
    if transactions.empty:
        return []
    if 'Product_Key' in transactions.columns:
        product_column = 'Product_Key'
    else:
        product_column = 'ProductCategory'
    customer_products = transactions[transactions['CustomerID'] == customer_id][product_column].unique()
    if len(customer_products) == 0:
        return transactions[product_column].value_counts().head(3).index.tolist()
    other = transactions[transactions['CustomerID'] != customer_id]
    cooccurrence = other[other[product_column].isin(customer_products)][product_column].value_counts()
    recommendations = [item for item in cooccurrence.index if item not in customer_products]
    # Filter by segment
    if segment == 'Premium':
        recommendations = [r for r in recommendations if r in ['Luxury', 'Premium']]  # assuming categories
    elif segment == 'Value':
        recommendations = [r for r in recommendations if r in ['Standard', 'Premium']]
    else:
        recommendations = [r for r in recommendations if r in ['Value', 'Standard']]
    return recommendations[:3]

def build_transaction_products(df: pd.DataFrame):
    if 'Product_Key' in df.columns:
        return df.copy()
    df = df.copy()
    df['ProductCategory'] = pd.cut(df['Amount'], bins=[-1, 200, 500, 1000, 9999], labels=['Value', 'Standard', 'Premium', 'Luxury'])
    return df

def load_models():
    scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
    rf_model = joblib.load(RF_PATH) if os.path.exists(RF_PATH) else None
    xgb_model = joblib.load(XGB_PATH) if os.path.exists(XGB_PATH) else None
    return scaler, rf_model, xgb_model

def save_models(scaler, rf_model, xgb_model):
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(rf_model, RF_PATH)
    joblib.dump(xgb_model, XGB_PATH)

@st.cache_data
def get_rfm_data(password):
    try:
        raw_df = load_raw_data(password)
        if raw_df.empty:
            return pd.DataFrame()
        rfm_df = calculate_rfm(raw_df)
        return rfm_df[['CustomerID', 'Recency', 'Frequency', 'Monetary']]
    except Exception as e:
        st.error(f"Erreur lors du calcul des données RFM: {e}")
        return pd.DataFrame()

if refresh_button:
    st.cache_data.clear()

with st.spinner('Chargement des données...'):
    raw_df = load_raw_data(password) if password else pd.DataFrame()
    if raw_df.empty:
        st.warning('Aucune donnée SQL en direct disponible. Affichage des données d\'exemple jusqu\'à ce que la connexion soit établie.')
        raw_df = create_sample_data()

    rfm_df = calculate_rfm(raw_df)
    rfm_scaled, scaler = scale_features(rfm_df)
    wcss, silhouette, cluster_labels, linkage_matrix = run_clustering(rfm_scaled)
    cluster_summary = cluster_profile(rfm_df, cluster_labels)
    top_cluster = get_top_cluster(rfm_df, cluster_labels)
    rfm_df['Class'] = np.where(cluster_labels == top_cluster, 1, 0)
    rf_model, xgb_model, X_test, y_test, rf_score, xgb_score, rf_grid, xgb_grid = train_models(rfm_scaled, rfm_df['Class'])
    save_models(scaler, rf_model, xgb_model)

    sales_ts = aggregate_sales(raw_df)
    product_df = build_transaction_products(raw_df)
    scaler, rf_model_saved, xgb_model_saved = load_models()

st.title('Tableau de Bord Sougui - Intelligence Data Science')

tabs = st.tabs(['📊 Résumé Exécutif', '🎯 Intelligence Client', '🤖 Moteur Prédictif', '📉 Prévision Marché'])

with tabs[0]:
    st.header('Résumé Exécutif')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric('Total Clients', len(rfm_df))
    with col2:
        st.metric('Ventes Totales', f'{raw_df["Amount"].sum():,.0f} €')
    with col3:
        st.metric('Score Silhouette', f'{silhouette:.3f}')
    
    st.subheader('Aperçu des Données')
    st.dataframe(rfm_df.head())
    
    st.subheader('Distribution des Segments')
    segment_counts = cluster_summary['Segment'].value_counts()
    fig = px.pie(values=segment_counts.values, names=segment_counts.index, title='Répartition des Segments Clients')
    st.plotly_chart(fig)

with tabs[1]:
    st.header('Intelligence Client')
    st.subheader('Clustering RFM')
    st.latex(r"$$M = \sum(\text{Montant})$$")
    st.write('Comparaison K-Means et Clustering Hiérarchique')
    st.metric('Score Silhouette', f'{silhouette:.3f}')
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Graphique Elbow - K-Means')
        fig, ax = plt.subplots()
        ax.plot(range(1, 11), wcss, marker='o')
        ax.set_title('Méthode Elbow')
        ax.set_xlabel('Nombre de Clusters')
        ax.set_ylabel('WCSS')
        st.pyplot(fig)
    
    with col2:
        st.subheader('Dendrogramme - Hiérarchique')
        dendro_fig = plot_dendrogram(linkage_matrix)
        st.pyplot(dendro_fig)
    
    st.subheader('Profilage des Clusters')
    st.dataframe(cluster_summary)
    
    st.subheader('Visualisation des Clusters')
    scatter_fig = px.scatter(
        rfm_scaled,
        x='Recency',
        y='Monetary',
        color=cluster_labels.astype(str),
        labels={'color': 'Cluster'},
        title='Clusters K-Means',
    )
    st.plotly_chart(scatter_fig, use_container_width=True)

with tabs[2]:
    st.header('Moteur Prédictif')
    if rf_model is None or xgb_model is None:
        st.warning('Les modèles ne sont pas encore entraînés. Actualisez les données d\'abord.')
    else:
        st.subheader('Prédiction de Segment Client')
        if password:
            rfm_data = get_rfm_data(password)
            if not rfm_data.empty:
                customer_ids = rfm_data['CustomerID'].tolist()
                selected_customer = st.selectbox('Sélectionnez un Client', customer_ids, key='customer_select')
                if selected_customer:
                    customer_row = rfm_data[rfm_data['CustomerID'] == selected_customer].iloc[0]
                    recency = int(customer_row['Recency'])
                    frequency = int(customer_row['Frequency'])
                    monetary = float(customer_row['Monetary'])
                    st.text_input('Récence (jours)', value=str(recency), disabled=True)
                    st.text_input('Fréquence (achats)', value=str(frequency), disabled=True)
                    st.text_input('Monétaire (€)', value=str(monetary), disabled=True)
                    if st.button('Prédire'):
                        input_df = pd.DataFrame([[recency, frequency, monetary]], columns=['Recency', 'Frequency', 'Monetary'])
                        input_scaled = scaler.transform(input_df)
                        pred_rf = rf_model.predict(input_scaled)[0]
                        pred_xgb = xgb_model.predict(input_scaled)[0]
                        label_rf = 'Premium' if pred_rf == 1 else 'Standard/Value'
                        label_xgb = 'Premium' if pred_xgb == 1 else 'Standard/Value'
                        st.success(f'Random Forest: {label_rf}')
                        st.success(f'XGBoost: {label_xgb}')
            else:
                st.warning("Aucune donnée RFM trouvée dans la base de données.")
        else:
            st.warning("Entrez le mot de passe SQL pour accéder aux données clients.")
        
        st.subheader('Comparaison des Modèles')
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        y_pred_rf = rf_model.predict(X_test)
        y_pred_xgb = xgb_model.predict(X_test)
        metrics = {
            'Modèle': ['Random Forest', 'XGBoost'],
            'Précision': [accuracy_score(y_test, y_pred_rf), accuracy_score(y_test, y_pred_xgb)],
            'Précision (Precision)': [precision_score(y_test, y_pred_rf), precision_score(y_test, y_pred_xgb)],
            'Rappel': [recall_score(y_test, y_pred_rf), recall_score(y_test, y_pred_xgb)],
            'F1-Score': [f1_score(y_test, y_pred_rf), f1_score(y_test, y_pred_xgb)]
        }
        metrics_df = pd.DataFrame(metrics)
        st.dataframe(metrics_df.round(4))
        
        f1_rf = f1_score(y_test, y_pred_rf)
        f1_xgb = f1_score(y_test, y_pred_xgb)
        winner = 'Random Forest' if f1_rf > f1_xgb else 'XGBoost' if f1_xgb > f1_rf else 'Égalité'
        st.metric('Gagnant (F1-Score)', winner)
        
        st.subheader('Courbe ROC Combinée')
        fig, ax = plt.subplots(figsize=(8, 6))
        y_proba_rf = rf_model.predict_proba(X_test)[:, 1]
        y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
        fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
        fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_proba_xgb)
        auc_rf = roc_auc_score(y_test, y_proba_rf)
        auc_xgb = roc_auc_score(y_test, y_proba_xgb)
        ax.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.2f})', color='blue')
        ax.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {auc_xgb:.2f})', color='orange')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('Taux de Faux Positifs')
        ax.set_ylabel('Taux de Vrais Positifs')
        ax.set_title('Courbes ROC Comparées')
        ax.legend()
        st.pyplot(fig)
        
        st.subheader('Importance des Caractéristiques')
        feature_names = ['Récence', 'Fréquence', 'Monétaire']
        rf_importances = get_feature_importances(rf_model, feature_names)
        xgb_importances = get_feature_importances(xgb_model, feature_names)

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            ax.barh(feature_names, rf_importances, color='blue')
            ax.set_title('Random Forest')
            ax.set_xlabel('Importance')
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots()
            ax.barh(feature_names, xgb_importances, color='orange')
            ax.set_title('XGBoost')
            ax.set_xlabel('Importance')
            if xgb_importances.sum() == 0:
                st.warning('Importance XGBoost indisponible ou nulle.')
            st.pyplot(fig)

with tabs[3]:
    st.header('Prévision Marché')
    st.write('Prévision des ventes avec SARIMA et Prophet.')
    if sales_ts.empty:
        st.warning('Aucune série temporelle de ventes disponible.')
    else:
        st.line_chart(sales_ts)
        valid, reason = is_series_valid(sales_ts['Sales'])
        if not valid:
            st.warning(f'Analyse temporelle partielle : {reason}')
        else:
            adf_stat, adf_p = adf_test(sales_ts['Sales'])
            if adf_stat is None:
                st.error(f'Test ADF indisponible : {adf_p}')
            else:
                st.subheader('Test ADF (Stationnarité)')
                st.write(f'Statistique ADF : {adf_stat:.4f}')
                st.write(f'Valeur-p : {adf_p:.4f}')
                if adf_p <= 0.05:
                    st.success('La série est stationnaire selon le test ADF.')
                else:
                    st.warning('La série n\'est pas stationnaire selon le test ADF. SARIMA utilisera une différenciation.')

        forecast_series, forecast_info = generate_forecast(sales_ts['Sales'])
        if forecast_series is not None:
            st.subheader('Prévision de Ventes - SARIMA')
            st.write(forecast_info)
            future_df = pd.DataFrame({'Prévision': forecast_series})
            st.line_chart(future_df)

        st.subheader('Évaluation des modèles')
        comparison, infos, message = compare_forecasts(sales_ts['Sales'])
        if comparison:
            comp_df = pd.DataFrame(comparison).T
            st.dataframe(comp_df.round(4))
            if infos:
                st.write('Informations modèles :')
                st.json(infos)
        else:
            st.info(message or 'Comparaison indisponible pour cette série.')

# Recommendation section - perhaps add to a tab or sidebar
st.sidebar.markdown('---')
st.sidebar.subheader('Recommandation Produit')
customer_id = st.sidebar.number_input('Client_Key', min_value=1, value=int(raw_df['CustomerID'].iloc[0] if not raw_df.empty else 1))
# Get segment for customer
customer_rfm = rfm_df[rfm_df['CustomerID'] == customer_id]
if not customer_rfm.empty:
    customer_cluster = cluster_labels[rfm_df.index[rfm_df['CustomerID'] == customer_id][0]]
    customer_segment = cluster_summary[cluster_summary['cluster'] == customer_cluster]['Segment'].values[0]
    st.sidebar.write(f'Segment: {customer_segment}')
    recs = recommendation_engine(product_df, customer_id, customer_segment)
    if len(recs) == 0:
        st.sidebar.info('Aucune recommandation disponible.')
    else:
        st.sidebar.write('Top 3 recommandations:')
        for idx, item in enumerate(recs, start=1):
            st.sidebar.write(f'{idx}. {item}')
else:
    st.sidebar.warning('Client non trouvé.')
import streamlit as st

# Title of the application
st.title("Model Evaluation App")

# Sidebar for user input
st.sidebar.header("User Input")

# Function for user input
def get_user_input():
    model_type = st.sidebar.selectbox("Select Model Type:", ("Linear Regression", "Random Forest", "SVM"))
    dataset = st.sidebar.file_uploader("Upload Dataset", type=["csv"])
    return model_type, dataset

model_type, dataset = get_user_input()

# Load data if a dataset is uploaded
if dataset is not None:
    import pandas as pd
    data = pd.read_csv(dataset)
    st.write(data)

# Placeholder for model evaluation
if data is not None:
    if model_type == "Linear Regression":
        # Add code for Linear Regression evaluation
        st.write("Linear Regression Model Evaluation")
    elif model_type == "Random Forest":
        # Add code for Random Forest evaluation
        st.write("Random Forest Model Evaluation")
    elif model_type == "SVM":
        # Add code for SVM evaluation
        st.write("SVM Model Evaluation")

# Run the application with 'streamlit run app.py' command

