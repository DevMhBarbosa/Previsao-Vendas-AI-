import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from prophet import Prophet
import matplotlib.pyplot as plt

# Simulação de um conjunto de dados de vendas
def generate_data():
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=1000, freq='D')
    sales = np.random.randint(100, 500, size=len(dates))
    return pd.DataFrame({'date': dates, 'sales': sales})

def prepare_prophet_data(df):
    df_prophet = df.rename(columns={'date': 'ds', 'sales': 'y'})
    print(df_prophet.head())  # Verifica se os dados foram corretamente formatados
    return df_prophet

def train_prophet(df_prophet):
    model = Prophet()
    model.fit(df_prophet)
    if model.history is None:
        print("Erro: O modelo não foi treinado corretamente.")
    else:
        print("Modelo treinado com sucesso!")
    return model

def make_forecast(model, periods=365):
    future = model.make_future_dataframe(periods=periods)
    return model.predict(future)

def train_random_forest(df):
    X = np.array([date.dayofyear for date in df['date']]).reshape(-1, 1)
    y = df['sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return X_test, y_test, predictions

def plot_prophet(model, forecast):
    fig = model.plot(forecast)
    plt.title("Previsão de Vendas com Prophet")
    plt.show()

def plot_random_forest(X_test, y_test, predictions):
    plt.figure(figsize=(10, 5))
    plt.scatter(X_test, y_test, color='blue', label='Valores Reais')
    plt.scatter(X_test, predictions, color='red', label='Previsões RandomForest')
    plt.legend()
    plt.title("Comparação de Previsões RandomForest")
    plt.show()

# Execução do fluxo de trabalho
df = generate_data()
df_prophet = prepare_prophet_data(df)
prophet_model = train_prophet(df_prophet)
forecast = make_forecast(prophet_model)
X_test, y_test, rf_predictions = train_random_forest(df)
plot_prophet(prophet_model, forecast)
plot_random_forest(X_test, y_test, rf_predictions)

print("Código executado com sucesso!")