import pandas as pd 
import os

NB_LIGNES = 100000

def extraire_par_note(df):
    df_temp = pd.DataFrame()
    for i in range(5):
        temp = df.loc[df["review/score"]== i + 1, :]
        df_temp = pd.concat([df_temp,temp.iloc[:NB_LIGNES]])
        print(f"score {i+1} extrait")
    return df_temp

def export_csv(df_a_extraire):
    if not os.path.exists("data"):
        os.mkdir("data")
    
    # --- MODIFICATION : Ajout du mélange (Shuffle) ---
    # frac=1 signifie qu'on prend 100% des lignes mais dans le désordre
    print("Mélange des données...")
    df_a_extraire = df_a_extraire.sample(frac=1, random_state=42).reset_index(drop=True)
    
    df_a_extraire.to_csv(f"data/books_rating_500k.csv",index = False)
    print(f"CSV extrait dans data")

if __name__ == "__main__":
    df = pd.read_csv("data/Books_rating.csv", sep= ",")
    df_resultat = extraire_par_note(df)
    
    export_csv(df_resultat)