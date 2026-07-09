import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def generate_forecast(df, date_column='Date'):
    """
    Retourne un dictionnaire contenant trois DataFrames :
        - 'historical' : toutes les semaines complètes (lundi-dimanche)
        - 'current_week' : dernière semaine complète + semaine en cours (partielle)
        - 'forecast' : dernière semaine complète + 4 semaines futures prédites
    """
    today = pd.Timestamp.now().normalize()
    df = df[df[date_column] <= today].copy()
    if df.empty:
        return None

    df[date_column] = pd.to_datetime(df[date_column]).dt.normalize()

    # Début de semaine (lundi) et fin (dimanche)
    df['week_start'] = df[date_column] - pd.to_timedelta(df[date_column].dt.weekday, unit='d')
    df['week_end'] = df['week_start'] + pd.Timedelta(days=6)

    # Semaines complètes : week_end < aujourd'hui
    df['is_complete'] = df['week_end'] < today
    complete_weeks = df[df['is_complete']]
    if complete_weeks.empty:
        return None

    # Dernière semaine complète
    last_complete_week = complete_weeks['week_start'].max()

    # --- Historique : toutes les semaines complètes ---
    hist = complete_weeks.groupby('week_start').size().reset_index(name='y')
    hist = hist.rename(columns={'week_start': 'ds'})
    hist['ds'] = pd.to_datetime(hist['ds']).dt.normalize()
    hist = hist.sort_values('ds').reset_index(drop=True)

    if len(hist) < 3:
        return None

    # Valeur de la dernière semaine complète
    last_value = hist[hist['ds'] == last_complete_week]['y'].iloc[0]

    # --- Semaine en cours (partielle) ---
    current_week_start = today - pd.Timedelta(days=today.weekday())
    current_count = len(df[df['week_start'] == current_week_start])

    current_week_df = pd.DataFrame({
        'ds': [last_complete_week, current_week_start],
        'y': [last_value, current_count]
    })

    # --- Prévision 4 semaines ---
    X = np.arange(len(hist)).reshape(-1, 1)
    y = hist['y'].values
    model = LinearRegression()
    model.fit(X, y)

    last_idx = len(hist) - 1
    future_indices = np.arange(last_idx + 1, last_idx + 5).reshape(-1, 1)
    preds = model.predict(future_indices)
    preds = np.maximum(preds, 0)  # pas de valeurs négatives

    future_dates = [last_complete_week + pd.Timedelta(weeks=i+1) for i in range(4)]

    forecast_df = pd.DataFrame({
        'ds': [last_complete_week] + future_dates,
        'y': [last_value] + preds.tolist()
    })

    return {
        'historical': hist,
        'current_week': current_week_df,
        'forecast': forecast_df
    }