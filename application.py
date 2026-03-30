import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
import os

# Force CPU-only mode (prevents GPU errors on free servers)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# -------------------------------
# TITLE
# -------------------------------
st.title("🦠 COVID-19 Prediction Dashboard")
st.write("Hybrid SEIRD + LSTM Model")

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/owid/covid-19-data/refs/heads/master/public/data/owid-covid-data.csv"
    df = pd.read_csv(url)
    df = df[df['location'] == 'India']
    df = df[['date', 'new_cases', 'population']].dropna()
    df = df[df['new_cases'] > 0]
    df.reset_index(drop=True, inplace=True)
    return df

df = load_data()
population = df['population'].iloc[0]
cases = df['new_cases'].values

# -------------------------------
# SIDEBAR CONTROLS
# -------------------------------
st.sidebar.header("⚙️ Controls")
beta = st.sidebar.slider("Transmission Rate (β)", 0.1, 1.0, 0.3)
gamma = st.sidebar.slider("Recovery Rate (γ)", 0.05, 0.5, 0.1)
mu = st.sidebar.slider("Mortality Rate (μ)", 0.001, 0.05, 0.01)

# -------------------------------
# SEIRD MODEL
# -------------------------------
def seird_model(y, t, beta, sigma, gamma, mu):
    S, E, I, R, D = y
    dS = -beta * S * I
    dE = beta * S * I - sigma * E
    dI = sigma * E - gamma * I - mu * I
    dR = gamma * I
    dD = mu * I
    return [dS, dE, dI, dR, dD]

I0 = cases[0] / population
E0 = I0 * 2
R0, D0 = 0, 0
S0 = 1 - (I0 + E0)
y0 = [S0, E0, I0, R0, D0]
t = np.arange(len(cases))
sigma = 0.2

solution = odeint(seird_model, y0, t, args=(beta, sigma, gamma, mu))
S, E, I, R, D = solution.T
seird_pred = I * population

# -------------------------------
# LSTM MODEL (OPTIMIZED FOR DEPLOYMENT)
# -------------------------------
@st.cache_resource
def load_or_train_model(cases, seq_len=14, epochs=5, batch_size=16):
    """
    Load pre-trained model if available, otherwise train and cache it.
    """
    model_path = 'covid_lstm_model.h5'
    
    # Try to load pre-trained model
    if os.path.exists(model_path):
        try:
            model = load_model(model_path)
            st.sidebar.success("✅ Loaded pre-trained model")
            return model
        except Exception as e:
            st.sidebar.warning(f"⚠️ Could not load model: {e}. Training new one...")
    
    # Fallback: train model (not recommended for production)
    scaler = MinMaxScaler()
    cases_scaled = scaler.fit_transform(cases.reshape(-1, 1))
    
    def create_sequences(data, seq_len=14):
        X, y = [], []
        for i in range(len(data)-seq_len):
            X.append(data[i:i+seq_len])
            y.append(data[i+seq_len])
        return np.array(X), np.array(y)
    
    X, y = create_sequences(cases_scaled)
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    
    model = Sequential([
        LSTM(50, input_shape=(seq_len, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    
    st.sidebar.info("🔄 Model trained on first load (consider saving it)")
    return model

# Load model
model = load_or_train_model(cases)

# Prepare data for prediction
scaler = MinMaxScaler()
cases_scaled = scaler.fit_transform(cases.reshape(-1, 1))

def create_sequences(data, seq_len=14):
    X, y = [], []
    for i in range(len(data)-seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

X, y = create_sequences(cases_scaled)
split = int(0.8 * len(X))
X_test = X[split:]

# Predict
pred_scaled = model.predict(X_test, verbose=0)
pred = scaler.inverse_transform(pred_scaled)
y_test_actual = scaler.inverse_transform(y[split:])

# -------------------------------
# ALIGN DATA
# -------------------------------
start_index = split + 14
seird_aligned = seird_pred[start_index : start_index + len(pred)]

y_true = y_test_actual.flatten()
lstm_pred = pred.flatten()
seird_pred = seird_aligned.flatten()

# -------------------------------
# HYBRID MODEL
# -------------------------------
alpha = 0.0  # LSTM performs best
hybrid_pred = alpha * seird_pred + (1 - alpha) * lstm_pred

# -------------------------------
# PLOTS
# -------------------------------
st.subheader("📊 Model Comparison")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(y_true, label="Real Data")
ax.plot(seird_pred, label="SEIRD")
ax.plot(lstm_pred, label="LSTM")
ax.plot(hybrid_pred, label="Hybrid", linewidth=3)
ax.legend()
ax.set_title("Prediction Comparison")
st.pyplot(fig)

# -------------------------------
# INSIGHTS
# -------------------------------
st.subheader("🧠 Insights")
st.write("""
- SEIRD captures overall trend but misses peaks  
- LSTM captures real-world fluctuations  
- Hybrid behaves similar to LSTM  
""")

# -------------------------------
# POLICY SIMULATION
# -------------------------------
st.subheader("🎛️ Policy Simulation")
lockdown = st.slider("Lockdown Strength (%)", 0, 100, 20)
effect = 1 - lockdown / 100
simulated_cases = lstm_pred * effect

fig2, ax2 = plt.subplots()
ax2.plot(lstm_pred, label="Original Prediction")
ax2.plot(simulated_cases, label="After Policy", linestyle='--')
ax2.legend()
ax2.set_title("Policy Impact Simulation")
st.pyplot(fig2)