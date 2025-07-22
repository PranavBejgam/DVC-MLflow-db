# import pandas as pd
# import xgboost as xgb
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import accuracy_score, precision_score, recall_score
# import mlflow 
# import mlflow.xgboost

# # loading data
# df = pd.read_csv('data/titanic.csv')

# # preprocess data
# df['Embarked'].fillna(df['Embarked'].mode()[0],inplace=True)
# df['Fare'].fillna(df['Fare'].median(),inplace=True)
# df = df.drop(['PassengerId','Name','Ticket','Cabin'],axis = 1,errors='ignore')
# le = LabelEncoder()
# df['Sex'] = le.fit_transform(df['Sex'])
# df['Embarked'] = le.fit_transform(df['Embarked'])


# # save preprocessed data
# df.to_csv('data/titanic_preprocessed.csv',index = False)

# # Split features and targets
# X = df.drop('Survived',axis=1)
# y = df['Survived']

# # train test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

# # Set mlflow experiment
# mlflow.set_experiment("Titanic XGBoost")

# # start mlflow.run
# with mlflow.start_run(run_name='With Age modified'):
#     # train XGBoost
#     model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
#     model.fit(X_train,y_train)

#     # predicta and evaluate
#     y_pred = model.predict(X_test)
#     accuracy = accuracy_score(y_test,y_pred)
#     precision = precision_score(y_test,y_pred)
#     recall = recall_score(y_test,y_pred)

#     # log parameters and metrics to mlflow
#     mlflow.log_param("n_estimators",100)
#     mlflow.log_param("max_depth",5)
#     mlflow.log_param("learning_rate",0.1)
#     mlflow.log_metric("Accuracy",accuracy)
#     mlflow.log_metric("Precision",precision)
#     mlflow.log_metric("Recall",recall)

#     # log model to mlflow
#     mlflow.xgboost.log_model(model,"model")

#     # Save model locally for DVC
#     model.save_model('models/xgboost_model.json')
#     print(f"Model trained and saved. Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")




# now fine tunning the code # scripts/process_and_train.py
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score
import mlflow
import mlflow.xgboost

# Load DVC-tracked data
df = pd.read_csv('data/titanic.csv')

# Preprocess data
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, errors='ignore')
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])

# Save preprocessed data
df.to_csv('data/titanic_preprocessed.csv', index=False)

# Split features and target
X = df.drop('Survived', axis=1)
y = df['Survived']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set MLflow experiment
mlflow.set_experiment("Titanic XGBoost")

# Define parameter grid for tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3]
}

# Start MLflow run
with mlflow.start_run(run_name="Version_2_Tuned"):
    # Perform grid search
    model = xgb.XGBClassifier(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get best model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # Log parameters and metrics to MLflow
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)

    # Log model to MLflow
    mlflow.xgboost.log_model(best_model, "model")

    # Save model locally for DVC
    best_model.save_model('models/xgboost_model_tuned.json')
    print(f"Tuned model saved. Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")