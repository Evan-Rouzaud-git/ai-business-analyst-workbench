import pandas as pd
import os

def load_dataset() :
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, '..', 'data', 'signalconso.csv')
    df = pd.read_csv(file_path, sep=';').copy()

    df = df.rename(columns={
        "id" : "record_id",
        "Signalement Lu par l'entreprise" : "is_read",
        "Signalement ayant reçu une réponse" : "has_response",
        "Signalement Transmis" : "transmis_authorite",
        "forwardToReponseConso" : "transmis_entreprise",
    })
    df.drop(columns=['contactAgreement','dep_name', 'dep_code', 'Code Officiel Région', 'Nom Officiel Région'], inplace=True)

    # on néttoie les valeurs binaires
    bin_col = ['transmis_entreprise', 'transmis_authorite', 'is_read','has_response']
    df[bin_col].fillna(0).astype(int)

    # on remplace dans tag nan par aucun tag
    df["tags"] = df["tags"].fillna("aucun_tag")

    # de meme pour category, status
    df["category"] = df["category"].fillna("aucune_categorie")
    df['status'] = df['status'].fillna("NonTraite")
    df['subcategories'] = df['subcategories'].fillna("aucune_sous_categorie")

    # on s'assure que la date soit au bon format
    df["Date"] = pd.to_datetime(df["Date"])

    # créer la colonne text pour du clustering avec categorie, subcategorie et tag
    def build_rows(rows):
        parts = []
        if pd.notna(rows["category"]) and rows["category"]!="aucune_categorie" :
            parts.append(str(rows["category"].replace("_"," ")))
        if pd.notna(rows["subcategories"]) and rows["subcategories"]!="aucune_sous_categorie" :
            parts.append(str(rows["subcategories"].replace("_"," ")))
        if pd.notna(rows["tags"]) and rows["tags"]!="aucun_tag" :
            parts.append(str(rows["tags"].replace("_"," ")))
        if not parts : 
            return "signalement consomateur sans description"
        return " ".join(parts)

    df["text"] = df.apply(build_rows, axis=1)
    return df

if __name__ == "__main__" : 
    df = load_dataset()
    print (df.describe)
    print(df.head())