# DVC-MLflow-db: Titanic Survival Prediction with DVC and MLflow

This project demonstrates a machine learning workflow for predicting Titanic passenger survival using a PostgreSQL database, DVC for data versioning, and MLflow for experiment tracking. The dataset is sourced from a PostgreSQL database, versioned with DVC, preprocessed, and used to train XGBoost models. Multiple data versions are created, models are trained and compared, and the best version (`v2-data`) is selected for further tuning.

## Project Overview

- **Objective**: Predict passenger survival using the Titanic dataset.
- **Data Source**: PostgreSQL database (`titanic_db`, table `titanic`).
- **Data Versions**:
  - `v1-data`: Original dataset with missing `Age` values.
  - `v2-data`: Missing `Age` values imputed with mean (best version, precision: 0.84, recall: 0.69).
  - `v3-data`: Added `FamilySize` feature (`SibSp + Parch + 1`).
- **Tools**:
  - **DVC**: Data and model versioning.
  - **MLflow**: Experiment tracking and model logging.
  - **XGBoost**: Classification model with hyperparameter tuning.
- **Workflow**: Import data from database, create versions, preprocess, train models, compare metrics in MLflow, select the best version, and tune the model.

## Prerequisites

- Python 3.8+
- PostgreSQL (with `titanic_db` database and `titanic` table)
- DVC (for data versioning)
- MLflow (for experiment tracking)
- Git (for version control)
- Required Python packages (see `requirements.txt`):
  ```plaintext
  pandas
  xgboost
  scikit-learn
  mlflow
  python-dotenv
  psycopg2-binary
  ```

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/PranavBejgam/DVC-MLflow-db.git
   cd DVC-MLflow-db
   ```

2. **Set Up Python Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Set Up PostgreSQL Database**:
   - Ensure PostgreSQL is running locally.
   - Create the `titanic_db` database and load the Titanic dataset into the `titanic` table.
   - Example SQL to create the table:
     ```sql
     CREATE TABLE titanic (
         "PassengerId" INTEGER,
         "Survived" INTEGER,
         "Pclass" INTEGER,
         "Name" TEXT,
         "Sex" TEXT,
         "Age" FLOAT,
         "SibSp" INTEGER,
         "Parch" INTEGER,
         "Ticket" TEXT,
         "Fare" FLOAT,
         "Cabin" TEXT,
         "Embarked" TEXT
     );
     ```
   - Import data (e.g., from a CSV) using `psql` or a tool like pgAdmin.

4. **Configure Environment Variables**:
   - Create a `.env` file in the project root:
     ```env
     DB_URL=postgresql://postgres:0021@localhost:5432/titanic_db
     ```
   - Ensure `.env` is in `.gitignore` to avoid committing credentials:
     ```gitignore
     .env
     ```

5. **Initialize DVC**:
   ```bash
   dvc init
   git add .dvc/config .dvc/.gitignore .dvcignore
   git commit -m "Initialize DVC"
   ```

6. **Set Up DVC Remote**:
   ```bash
   dvc remote add -d mystorage ./localstorage
   git add .dvc/config
   git commit -m "Add DVC remote storage"
   ```

## Workflow

### 1. Import Data from Database
Use `scripts/dvc_import.py` to import the `titanic` table:
```bash
python scripts/dvc_import.py
git add data/titanic.csv.dvc
git commit -m "Import initial Titanic data"
git tag v1-data
git push origin main
git push origin v1-data
dvc push
```

### 2. Create Data Versions
- **Version 2: Impute Missing Age with Mean**:
  ```bash
  psql -U postgres -d titanic_db
  UPDATE titanic SET "Age" = (SELECT AVG("Age") FROM titanic WHERE "Age" IS NOT NULL) WHERE "Age" IS NULL;
  \q
  python scripts/dvc_import.py
  git add data/titanic.csv.dvc
  git commit -m "Version 2: Impute missing Age with mean"
  git tag v2-data
  git push origin main
  git push origin v2-data
  dvc push
  ```

- **Version 3: Add FamilySize Feature**:
  ```bash
  psql -U postgres -d titanic_db
  ALTER TABLE titanic ADD COLUMN "FamilySize" INTEGER;
  UPDATE titanic SET "FamilySize" = "SibSp" + "Parch" + 1;
  \q
  python scripts/dvc_import.py
  git add data/titanic.csv.dvc
  git commit -m "Version 3: Add FamilySize feature"
  git tag v3-data
  git push origin main
  git push origin v3-data
  dvc push
  ```

### 3. Train Models on Each Version
Create the `models/` folder:
```bash
mkdir models
```

Train on each version, updating `run_name` in `scripts/process_and_train.py`:
- **Version 1**:
  ```bash
  git checkout v1-data
  dvc checkout
  python scripts/process_and_train.py  # Set run_name="Version_1"
  dvc add data/titanic_preprocessed.csv
  dvc add models/xgboost_model.json
  git add data/titanic_preprocessed.csv.dvc models/xgboost_model.json.dvc
  git commit -m "Train model on Version 1 data"
  git tag v1-model
  git push origin main
  git push origin v1-model
  dvc push
  ```

- **Version 2 (best model, precision: 0.84, recall: 0.69)**:
  ```bash
  git checkout v2-data
  dvc checkout
  python scripts/process_and_train.py  # Set run_name="Version_2"
  dvc add data/titanic_preprocessed.csv
  dvc add models/xgboost_model.json
  git add data/titanic_preprocessed.csv.dvc models/xgboost_model.json.dvc
  git commit -m "Train model on Version 2 data"
  git tag v2-model
  git push origin main
  git push origin v2-model
  dvc push
  ```

- **Version 3**:
  ```bash
  git checkout v3-data
  dvc checkout
  python scripts/process_and_train.py  # Set run_name="Version_3_with_FamilySize"
  dvc add data/titanic_preprocessed.csv
  dvc add models/xgboost_model.json
  git add data/titanic_preprocessed.csv.dvc models/xgboost_model.json.dvc
  git commit -m "Train model on Version 3 data"
  git tag v3-model
  git push origin main
  git push origin v3-model
  dvc push
  ```

### 4. Compare Models in MLflow
Start MLflow UI:
```bash
mlflow ui
```
Open `http://localhost:5000`, navigate to the `Titanic_XGBoost` experiment, and compare runs (`Version_1`, `Version_2`, `Version_3_with_FamilySize`). The best model is `Version_2` (precision: 0.84, recall: 0.69).

### 5. Select Best Data Version
Switch to `v2-data`:
```bash
git checkout v2-data
dvc checkout
```

Verify data:
```bash
python scripts/check_data.py
```

### 6. Train Tuned Model
Run the tuned model on `v2-data`:
```bash
python scripts/process_and_train.py  # Set run_name="Version_2_Tuned"
dvc add data/titanic_preprocessed.csv
dvc add models/xgboost_model_tuned.json
git add data/titanic_preprocessed.csv.dvc models/xgboost_model_tuned.json.dvc
git commit -m "Train tuned XGBoost model on Version 2 data"
git tag v2-tuned-model
git push origin main
git push origin v2-tuned-model
dvc push
```

### 7. Deploy Model (Optional)
Serve the tuned model:
```bash
mlflow models serve -m runs:/<run_id>/model --port 1234
```
Replace `<run_id>` with the `Version_2_Tuned` run ID from MLflow UI.

## Usage
- **Check Data**: `python scripts/check_data.py`
- **Import Data**: `python scripts/dvc_import.py`
- **Train Model**: `python scripts/process_and_train.py`
- **View Metrics**: `mlflow ui` and open `http://localhost:5000`
- **Switch Versions**: `git checkout <tag> && dvc checkout` (e.g., `v2-data`)

## Notes
- Ensure the DVC remote (`./localstorage`) is accessible.
- Use `.env` for secure database credentials.
- The best model (`v2-model`) was chosen for its high precision (0.84) and acceptable recall (0.69).
- For further experimentation, try `scripts/process_and_train_rf.py` for a Random Forest model.

