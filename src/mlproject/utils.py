import pymysql
import os
import sys
from sqlalchemy import create_engine  #  Needed for pandas
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
import pandas as pd
from dotenv import load_dotenv
from urllib.parse import quote_plus

load_dotenv()
host = os.getenv("host")
user = os.getenv("user")
password = os.getenv("password")
db = os.getenv("db")



def read_sql_data():
    logging.info("Reading SQL database started")
    try:
        escaped_password = quote_plus(password)
        # ✅ Use SQLAlchemy engine instead of raw pymysql connection
        engine = create_engine(
            f"mysql+pymysql://{user}:{escaped_password}@{host}/{db}"
        )

        logging.info(f"CONNECTION ESTABLISHED {engine}")  # ✅ Fixed logging syntax

        df = pd.read_sql_query('SELECT * FROM student', engine)
        print(df.head())
        return df

    except Exception as ex:
        raise CustomException(ex, sys)


