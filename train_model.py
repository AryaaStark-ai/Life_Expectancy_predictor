import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import joblib
from scipy.sparse import issparse

# Constants
TARGET_COLUMN = 'Life expectancy'
MODEL_SAVE_PATH = 'models/life_expectancy_model.h5'
PIPELINE_SAVE_PATH = 'models/preprocessing_pipeline.pkl'
DATA_PATH = 'data/final.csv'

# Load the dataset
df = pd.read_csv(DATA_PATH)

# Separate features and target
X = df.drop(TARGET_COLUMN, axis=1)
y = df[TARGET_COLUMN].values

# Identify categorical and numerical columns
categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()

# Preprocessing for numeric and categorical data
numeric_transformer = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=5)),  # Handle missing values
    ('scaler', RobustScaler())  # Scale numerical features
])
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)  # Encode categorical features

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_columns),
        ('cat', categorical_transformer, categorical_columns)
    ]
)

# Create a preprocessing pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# Fit and transform the features
X_preprocessed = pipeline.fit_transform(X)

# Convert sparse matrix to dense if necessary
if issparse(X_preprocessed):
    X_preprocessed = X_preprocessed.toarray()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

# Build the neural network model
model = Sequential([
    Dense(128, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.2),  # Regularization to prevent overfitting
    Dense(64, activation='relu'),
    Dropout(0.1),
    Dense(1, activation='linear')  # Output layer for regression
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Save the trained model and preprocessing pipeline
model.save(MODEL_SAVE_PATH)
joblib.dump(pipeline, PIPELINE_SAVE_PATH)

# Evaluate the model on the test set
loss, mae = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Loss: {loss:.4f}, Test MAE: {mae:.4f}")
