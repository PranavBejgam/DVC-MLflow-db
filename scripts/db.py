import pandas as pd
from sqlalchemy import create_engine

# Load CSV
df = pd.read_csv('data/titanic.csv')

# Create database connection
engine = create_engine('postgresql://postgres:0021@localhost:5432/titanic_db')
with engine.connect() as conn:
    print("Connection successful!")

# Import to database
df.to_sql('titanic', engine, if_exists='replace', index=False)
print("Data imported to database successfully!")