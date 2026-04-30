import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def log(message):
    print(f"[INFO]{message}")

def error(message):
    print(f"[ERROR]{message}")

file_path="data/creditcard.csv"

try :
    if not os.path.exists(file_path):
         raise FileNotFoundError("Dataset File not Found")
    
    df=pd.read_csv(file_path)
    log("Dataset Loaded successfully")

except Exception as e:
    error(f"Dataset not loaded :{e}")
    exit()

try:
    if df.empty:
        raise ValueError("Dataset is empty")
    log(f"Dataset shape :{df.shape}")
    log(f"Dataset columns:{list(df.columns)}")

except Exception as e :
    error(e)
    exit()

TARGET="Class"
try:
    if TARGET not in df.columns:
        raise ValueError(f"Column {TARGET} not found")
    
except Exception as e:
    exit()

try:
    missing=df.isnull().sum().sum()
    log(f"Total missing values {missing}")

    if missing>0:
        log("Remove missing Values")
        df=df.dropna

except Exception as e:
    error(f"Error Handling missing values {e}")

try:
    fraud=df[df[TARGET]==1]
    normal=df[df[TARGET]==0]
    log(f"Fraud Count :{len(fraud)}")
    log(f"Normal Count:{len(normal)}")

    if len(fraud)==0 or len(normal)==0:
        raise ValueError("Class is empty")
    
    sample_size=min(len(fraud),len(normal))
    normal_sample=normal.sample(n=sample_size,random_state=42)

    df_balanced=pd.concat([fraud,normal_sample])
    df_balanced=df_balanced.sample(frac=1,random_state=42)
    log("Class Balancing Completed")

except Exception as e:
    error(f"Error in class balancing {e}")
    exit()


try:
    X=df_balanced.drop(TARGET,axis=1)
    y=df_balanced[TARGET]
    
    if X.empty:
        raise ValueError("Feature set is empty")
    
except Exception as e:
    error(f"Error in Feature Split {e}")
    exit()

try:
    X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    log("Train-Test-Split successful")

except Exception as e:
    error(f"Error in Train-Test-Split :{e}")
    exit()

try:
    scaler=StandardScaler()
    X_train_scaled=scaler.fit_transform(X_train)
    X_test_scaled=scaler.transform(X_test)

    X_train_scaled=pd.DataFrame(X_train_scaled,columns=X.columns)
    X_test_scaled=pd.DataFrame(X_test_scaled,columns==X.columns)

    log("Feature scaling completed")

except Exception as e:
    error(f"Error in scaling :{e}")
    exit()


try:
    od.makedirs("data/processed",exist_ok=True)
    X_train_scaled.to_csv("data/preprocessed/X_train.csv",index=False)
    X_test_scaled.to_csv("data/preprocessed/X_test.csv",index=False)
    y_train.to_csv("data/preprocessed/y_train.csv",index=False)
    y_test.to_csv("data/preprocessed/y_test.csv",index=False)

    log("Processed data saved successfully")

except Exception as e:
    error(f"Error saving processed data {e}")
    exit()

try:
    os.makedirs("models",exist_ok=True)
    joblib.dump(scaler,"models/scaler.pkl")
    log("Scaler saved successfully")

except Exception as e:
    error(f"Error saving scaler :{e}")
    exit()

log("Preprocessing completed")
    
    


