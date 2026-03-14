import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib

# Load your data (change filename/columns if needed)
df = pd.read_csv("self_compacting_concrete.csv")
X = df[['Cement', 'FlyAsh', 'Water', 'Superplasticizer', 'CoarseAggregate', 'FineAggregate']]  # add GGBS etc. if present
y = df['CompressiveStrength']   # or 'SlumpFlow' etc.

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build ANN (feel free to change layers/neurons)
model = Sequential([
    Dense(64, activation='relu', input_shape=(X.shape[1],)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1)   # single output (strength in MPa)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=200, batch_size=16, validation_split=0.2, verbose=1)

# Save model + scaler
model.save("scc_ann_model.h5")
joblib.dump(scaler, "scaler.pkl")

# Quick evaluation
from sklearn.metrics import r2_score
pred = model.predict(X_test)
print("R² Score:", r2_score(y_test, pred))  # Usually 0.85–0.95+ for good SCC data
