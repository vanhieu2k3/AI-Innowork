import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
import joblib

df = pd.read_csv("children_health_data_update.csv")
print("Dữ liệu 5 hàng đầu:")
print(df.head())

x = df.drop('outcome', axis=1)  
y = df['outcome'] 

numerical_features = x.select_dtypes(include=['float64', 'int64']).columns
categorical_features = x.select_dtypes(include=['object']).columns

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),  
        ('cat', OneHotEncoder(), categorical_features)  
    ])

x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=42)
x_train_preprocessed = preprocessor.fit_transform(x_train)
x_test_preprocessed = preprocessor.transform(x_test)

param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto'],
    'kernel': ['linear', 'rbf']
}

grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(x_train_preprocessed, y_train)
svm_model = grid_search.best_estimator_

y_pred = svm_model.predict(x_test_preprocessed)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

joblib.dump(svm_model, 'svm_model.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
