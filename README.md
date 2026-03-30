Hybrid SEIRD-LSTM COVID-19 Prediction Dashboard
A hybrid epidemiological and deep learning framework for modeling and forecasting COVID-19 trends in India, complete with an interactive policy simulation dashboard.

Quick Overview
This project blends classic SEIRD compartmental modeling (Susceptible-Exposed-Infected-Recovered-Deceased) with LSTM neural networks for time-series forecasting. The result? A hybrid prediction system powered by an intuitive Streamlit dashboard for visualizations and "what-if" policy experiments. It bridges rigorous math-based epidemic modeling with data-driven ML to boost accuracy and support smarter decision-making.

Standout Features
Real-time COVID-19 data viz for India

SEIRD epidemiological simulations

LSTM-powered time-series forecasts

Hybrid model fusing SEIRD + LSTM

Policy simulators (e.g., lockdown effects)

Fully interactive Streamlit dashboard

How It Works
1. SEIRD Model
Built on differential equations, it nails the big-picture trends in disease spread—like how infections build and fade over time.

2. LSTM Model
This deep learning beast learns from real data to spot peaks, dips, and quirky fluctuations that pure math might miss.

3. Hybrid Magic
We combine them smartly:
Hybrid Prediction = α × SEIRD + (1 − α) × LSTM
(It's currently tilted toward LSTM for sharper short-term predictions, but tunable.)

Data & Tech
Dataset: Our World in Data (OWID) for India—key fields like date, new_cases, and population.

Stack: Python, Streamlit (dashboard), TensorFlow/Keras (LSTM), Scikit-learn (preprocessing), SciPy (ODE solver), Matplotlib, Pandas/NumPy.

Get It Running
Clone the repo:
git clone https://github.com/kssandraeshwar/Hybrid-SEIRD-LSTM-COVID-Prediction.git
cd Hybrid-SEIRD-LSTM-COVID-Prediction

Install deps: pip install -r requirements.txt

Launch: streamlit run application.py

Head to http://localhost:8501

Dashboard Highlights
Expect model comparison charts (SEIRD vs. LSTM vs. Hybrid), real-vs-predicted overlays, lockdown strength sliders, and tweakable params like β (transmission), γ (recovery), and μ (mortality).

What We've Learned
SEIRD shines for long-term trends.

LSTM handles real-world noise.

The hybrid strikes a sweet spot between explainability and precision.

Simulations reveal how interventions bend case curves.

Next Steps
Auto-optimize α with validation.

Layer in XGBoost, GRU, or Transformers.

Go multi-country.

Hook up live APIs.

Add confidence intervals for uncertainty.

Why It Matters
Perfect for epi modeling, time-series ML, AI in healthcare, or policy sims. Great for conference papers, theses, or your research portfolio.
