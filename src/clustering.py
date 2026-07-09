# src/clustering.py
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from collections import Counter

class ClusteringEngine:
    def __init__(self, n_macro=6, n_clusters_fin=30, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.n_macro = n_macro
        self.n_clusters_fin = n_clusters_fin
        self.model_name = model_name
        self._model = None
        self._nlp = None
        self.embeddings = None
        self.embeddings_normalized = None
        self.labels_fin = None
        self.macro_labels = None
        self.final_macro_labels = None
        self.noms_sujets = {}
        self.cluster_info = []
        self.macro_counts = {}

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _get_nlp(self, progress_callback=None):
        if self._nlp is None:
            try:
                import spacy
                try:
                    self._nlp = spacy.load("fr_core_news_sm")
                except OSError:
                    if progress_callback:
                        progress_callback("📦 Installation du modèle linguistique...", 0.82)
                    import subprocess
                    subprocess.run(["python", "-m", "spacy", "download", "fr_core_news_sm"], check=True, capture_output=True)
                    if progress_callback:
                        progress_callback("✅ Modèle linguistique installé", 0.85)
                    self._nlp = spacy.load("fr_core_news_sm")
            except ImportError:
                raise ImportError("spacy n'est pas installé. pip install spacy")
        return self._nlp

    def fit(self, df, text_column='description', progress_callback=None):
        from sklearn.preprocessing import normalize
        from sklearn.cluster import AgglomerativeClustering

        sentences = df[text_column].fillna('').astype(str).values

        # Étape 1 : Chargement du modèle
        if progress_callback:
            progress_callback("🧠 Chargement du modèle...", 0.0)
        model = self._get_model()

        # Étape 2 : Embeddings par lots (progression 0.05 → 0.65)
        batch_size = 64
        n = len(sentences)
        embeddings_list = []
        
        # Afficher un message global plutôt que par batch pour moins de bruit
        if progress_callback:
            progress_callback("🧠 Génération des embeddings...", 0.05)
        
        total_batches = (n + batch_size - 1) // batch_size
        for i in range(0, n, batch_size):
            batch = sentences[i:i+batch_size]
            batch_emb = model.encode(batch, show_progress_bar=False)
            embeddings_list.append(batch_emb)
            
            # Mise à jour moins fréquente pour éviter le "bruit"
            if progress_callback and i % (batch_size * 4) == 0:  # Tous les 4 batches
                progress_val = 0.05 + 0.6 * (min(i + batch_size, n) / n)
                progress_callback(f"🧠 Embeddings {min(i + batch_size, n)}/{n}", progress_val)
        
        self.embeddings = np.vstack(embeddings_list)
        self.embeddings_normalized = normalize(self.embeddings, norm='l2')

        # Étape 3 : Clustering fin (0.65 → 0.75)
        if progress_callback:
            progress_callback("🔍 Regroupement des sujets similaires...", 0.65)
        agg = AgglomerativeClustering(n_clusters=self.n_clusters_fin, metric='euclidean', linkage='ward')
        self.labels_fin = agg.fit_predict(self.embeddings_normalized)

        # Centroïdes
        centroids = []
        for label in range(self.n_clusters_fin):
            indices = np.where(self.labels_fin == label)[0]
            if len(indices) > 0:
                centroids.append(np.mean(self.embeddings_normalized[indices], axis=0))
            else:
                centroids.append(np.zeros(self.embeddings_normalized.shape[1]))
        centroids = np.array(centroids)

        # Étape 4 : Macro‑clustering (0.75 → 0.80)
        if progress_callback:
            progress_callback("📊 Identification des thèmes principaux...", 0.75)
        macro_clustering = AgglomerativeClustering(n_clusters=self.n_macro, metric='cosine', linkage='average')
        self.macro_labels = macro_clustering.fit_predict(centroids)
        self.final_macro_labels = np.array([self.macro_labels[label] for label in self.labels_fin])

        # Étape 5 : Noms de sujets (0.80 → 0.95)
        if progress_callback:
            progress_callback("🏷️ Génération des noms de thèmes...", 0.80)
        nlp = self._get_nlp(progress_callback=progress_callback)
        self._generate_topic_names(sentences, nlp)
        self._build_cluster_info(df, sentences, nlp)

        if progress_callback:
            progress_callback("✅ Clustering terminé !", 0.95)

        return self.final_macro_labels

    def _generate_topic_names(self, sentences, nlp):
        macro_texts = {}
        for macro_id in range(self.n_macro):
            indices = np.where(self.final_macro_labels == macro_id)[0]
            if len(indices) > 0:
                sample_indices = indices[:min(20, len(indices))]
                sample_texts = [sentences[i] for i in sample_indices]
                macro_texts[macro_id] = " ".join(sample_texts)
            else:
                macro_texts[macro_id] = ""
        for label, text in macro_texts.items():
            if text.strip():
                doc = nlp(text)
                mots_cles = [
                    token.lemma_ for token in doc
                    if token.pos_ in ("NOUN", "PROPN") and len(token.text) > 2
                ]
                top_mots = Counter(mots_cles).most_common(3)
                self.noms_sujets[label] = " ".join([mot for mot, _ in top_mots])
            else:
                self.noms_sujets[label] = f"Thème {label}"

    def _build_cluster_info(self, df, sentences, nlp):
        self.cluster_info = []
        self.macro_counts = Counter(self.final_macro_labels)
        for macro_id in range(self.n_macro):
            indices = np.where(self.final_macro_labels == macro_id)[0]
            if len(indices) == 0:
                continue
            clusters_in_macro = set(self.labels_fin[indices])
            examples = [sentences[i] for i in indices[:5]]
            categories = {}
            if 'category' in df.columns:
                for idx in indices:
                    cat = df.iloc[idx]['category']
                    if pd.notna(cat):
                        categories[cat] = categories.get(cat, 0) + 1
            all_words = []
            for idx in indices[:100]:
                text = sentences[idx]
                doc = nlp(text)
                words = [token.lemma_ for token in doc if token.pos_ in ("NOUN", "PROPN", "ADJ") and len(token.text) > 2]
                all_words.extend(words)
            top_words = [w for w, _ in Counter(all_words).most_common(8)]
            self.cluster_info.append({
                'cluster_id': macro_id,
                'count': len(indices),
                'share': len(indices) / len(sentences) * 100,
                'indices': indices,
                'examples': examples,
                'categories': dict(sorted(categories.items(), key=lambda x: -x[1])[:5]),
                'top_words': top_words,
                'clusters_in_macro': sorted(clusters_in_macro)
            })

    def get_cluster_info(self):
        return self.cluster_info

    def get_macro_counts(self):
        return self.macro_counts