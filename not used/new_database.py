import pandas as pd
from deta import Deta
import pandas as pd
import datetime

def insert_from_csv(input_dict, db):
      # insert a new deck to the database
      db.put(input_dict)
      
df = pd.read_csv("db.csv", index_col=False)
key = "a00bf83l_ySa7Affs5ghqdQPp5w3Yri2Xuw51gGV9"

# 2) initialize with a project key
deta = Deta(key)

# 3) create and use as many DBs as you want!
db = deta.Base("ygo_elo_database")

cols = df.columns
for i in range(len(df)):
      input_dict = {}
      for c in cols:
            if c=='History':
                  input_dict[c] = dict(eval(df.at[i, c]))
            else:
                input_dict[c] = str(df.at[i, c])
      insert_from_csv(input_dict, db)