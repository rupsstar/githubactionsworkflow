import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
column_names = [
    "symboling", "normalized-losses", "make", "fuel-type", "aspiration",
    "num-of-doors", "body-style", "drive-wheels", "engine-location",
    "wheel-base", "length", "width", "height", "curb-weight", "engine-type",
    "num-of-cylinders", "engine-size", "fuel-system", "bore", "stroke",
    "compression-ratio", "horsepower", "peak-rpm", "city-mpg", "highway-mpg", "price"
]

df = pd.read_csv(url, names=column_names, na_values="?")

# Drop rows with missing target (fuel-type)
df.dropna(subset=['fuel-type'], inplace=True)
# Fill missing numerical values with column means
for column in df.select_dtypes(include=['float64', 'int64']).columns:
    df[column].fillna(df[column].mean(), inplace=True)

# Encode categorical features
label_encoders = {}
categorical_features = ['make', 'aspiration', 'body-style', 'drive-wheels', 
                        'engine-location',
                        'engine-type', 'fuel-system', 'num-of-cylinders']
for feature in categorical_features:
    le = LabelEncoder()
    df[feature] = le.fit_transform(df[feature])
    label_encoders[feature] = le

# Encode target variable
le_fuel = LabelEncoder()
df['fuel-type'] = le_fuel.fit_transform(df['fuel-type'])

# Select features and target
features = [
    "wheel-base", "length", "width", "height", "curb-weight",
    "engine-size", "horsepower", "peak-rpm", "city-mpg", "highway-mpg"
]

X = df[features]
y = df['fuel-type']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define parameter grid for GridSearchCV
param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

# Initialize the model
knn = KNeighborsClassifier()

# Set up the GridSearchCV
grid_search = GridSearchCV(estimator=knn, 
                           param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Fit the model using GridSearchCV
grid_search.fit(X_train, y_train)

# Best parameters and accuracy
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Best Parameters: {best_params}")
print(f"Test Accuracy: {accuracy:.2f}")
