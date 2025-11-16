# 🚀 Setup Guide - NER Immobilier Bilingue

Guide complet pour cloner, installer et utiliser le modèle NER.

## 📋 Prérequis

- Python 3.7 ou supérieur
- pip (gestionnaire de paquets Python)
- 2 GB d'espace disque libre
- Connexion Internet pour télécharger les dépendances

---

## 🔧 Installation Complète

### Étape 1: Cloner le Repository
```bash
git clone https://github.com/your-username/NLP-urbanova.git
cd NLP-urbanova
```

### Étape 2: Créer un Environnement Virtuel

**Sur Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Sur macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Étape 3: Installer les Dépendances
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Installation complète (avec toutes les dépendances):**
```bash
pip install spacy pandas jsonlines tqdm
```

---

## 📦 Télécharger le Modèle Pré-Entraîné

Le modèle est trop volumineux pour GitHub. Deux options:

### Option A: Télécharger depuis Hugging Face 🤗 (RECOMMANDÉ)
```bash
pip install huggingface_hub
```
```python
from huggingface_hub import snapshot_download

# Télécharger le modèle
snapshot_download(
    repo_id="your-username/ner-immobilier-bilingue",
    local_dir="./output_model_immo_ner_bilingual_v3"
)
```

### Option B: Télécharger depuis Google Drive
```bash
# Télécharger le fichier depuis le lien fourni
# Lien: https://drive.google.com/file/d/YOUR_FILE_ID

# Extraire le modèle
unzip model-best.zip -d output_model_immo_ner_bilingual_v3/
```

### Option C: Entraîner le Modèle Vous-Même

Si vous voulez entraîner le modèle depuis zéro (20-40 minutes):
```bash
# 1. Générer les annotations
python 1_annotate_data.py

# 2. Préparer les données d'entraînement
python 2_train_model.py

# 3. Entraîner le modèle
python -m spacy train config_bilingual_fixed.cfg \
    --output output_model_immo_ner_bilingual_v3 \
    --paths.train train_bilingual_V3.spacy \
    --paths.dev dev_bilingual_V3.spacy
```

---

## ✅ Vérifier l'Installation
```bash
python -c "import spacy; print(spacy.__version__)"
```

Devrait afficher: `3.7.0` ou supérieur

---

## 🎯 Utilisation Rapide

### Test Simple
```python
import spacy

# Charger le modèle
nlp = spacy.load("output_model_immo_ner_bilingual_v3/model-best")

# Tester avec une phrase
text = "Appartement 3 chambres à louer Tunis 120m² 800 TND"
doc = nlp(text)

# Afficher les entités détectées
for ent in doc.ents:
    print(f"{ent.text:20} → {ent.label_}")
```

**Résultat attendu:**
```
Appartement          → TYPE
3 chambres           → BEDS
louer                → TRANSACTION
Tunis                → LOCATION
120m²                → AREA
800 TND              → PRICE
```

### Tester avec le Script de Test
```bash
python 3_test_model.py
```

Cela générera:
- Des statistiques de performance
- Des visualisations HTML dans `test_results/`
- Un rapport JSON détaillé

---

## 📊 Exemples d'Utilisation

### Exemple 1: Analyser une Seule Annonce
```python
import spacy

nlp = spacy.load("output_model_immo_ner_bilingual_v3/model-best")

# Texte en français
text_fr = """
Villa de luxe à vendre La Marsa. 5 chambres, 4 salles de bain,
superficie 350 m² avec piscine et jardin. Prix: 950000 TND.
État neuf, garage 2 places.
"""

doc = nlp(text_fr)

# Extraire les informations structurées
entities = {}
for ent in doc.ents:
    if ent.label_ not in entities:
        entities[ent.label_] = []
    entities[ent.label_].append(ent.text)

print(entities)
```

**Output:**
```python
{
    'TYPE': ['Villa'],
    'CONDITION': ['luxe', 'neuf'],
    'TRANSACTION': ['vendre'],
    'LOCATION': ['La Marsa'],
    'BEDS': ['5 chambres'],
    'BATHS': ['4 salles de bain'],
    'AREA': ['350 m²'],
    'AMENITY': ['piscine', 'jardin'],
    'PRICE': ['950000 TND'],
    'GARAGE': ['garage 2 places']
}
```

### Exemple 2: Traiter un Fichier CSV
```python
import spacy
import pandas as pd

# Charger le modèle
nlp = spacy.load("output_model_immo_ner_bilingual_v3/model-best")

# Charger vos annonces
df = pd.read_csv("mes_annonces.csv")

# Fonction d'extraction
def extract_entities(text):
    doc = nlp(text)
    entities = {}
    for ent in doc.ents:
        entities[ent.label_] = ent.text
    return entities

# Appliquer sur toutes les lignes
df['entities'] = df['description'].apply(extract_entities)

# Sauvegarder
df.to_csv("annonces_annotées.csv", index=False)
```

### Exemple 3: API REST Simple
```python
from flask import Flask, request, jsonify
import spacy

app = Flask(__name__)
nlp = spacy.load("output_model_immo_ner_bilingual_v3/model-best")

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.json['text']
    doc = nlp(text)
    
    entities = []
    for ent in doc.ents:
        entities.append({
            'text': ent.text,
            'label': ent.label_,
            'start': ent.start_char,
            'end': ent.end_char
        })
    
    return jsonify({'entities': entities})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

**Utilisation:**
```bash
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Appartement 3 chambres à louer Tunis 800 TND"}'
```

---

## 🐛 Dépannage

### Problème: `ModuleNotFoundError: No module named 'spacy'`

**Solution:**
```bash
pip install spacy pandas jsonlines tqdm
```

### Problème: `OSError: Can't find model`

**Solution:**
```bash
# Vérifier que le modèle existe
ls output_model_immo_ner_bilingual_v3/model-best/

# Si absent, télécharger ou entraîner le modèle (voir section ci-dessus)
```

### Problème: `ValueError: dictionary update sequence element`

**Solution:** Utilisez `config_bilingual_fixed.cfg` au lieu de l'ancien config.

### Problème: Entraînement très lent

**Solution:**
- C'est normal sur CPU (20-40 minutes)
- Réduire `max_steps` à 10000 dans le config pour un entraînement plus rapide
- Utiliser un GPU si disponible

### Problème: Faible taux de détection après entraînement

**Solution:**
```bash
# Vérifier le score ENTS_F dans l'output d'entraînement
# Devrait être > 95%

# Si < 95%, augmenter max_steps:
# Dans config_bilingual_fixed.cfg, changer:
# max_steps = 30000  # au lieu de 20000
```

---

## 📈 Performance Attendue

| Métrique | Valeur |
|----------|--------|
| Taux de détection | 92-95% |
| Temps de traitement | < 50ms par requête |
| Exemples d'entraînement | 3907 |
| Langues supportées | Français, Anglais |
| Entités reconnues | 10 types |

---

## 🔄 Mise à Jour du Modèle

Pour améliorer le modèle avec vos propres données:
```bash
# 1. Ajouter vos exemples dans 1_annotate_data.py
#    Section BILINGUAL_EXAMPLES

# 2. Ré-exécuter le pipeline complet
python 1_annotate_data.py
python 2_train_model.py
python -m spacy train config_bilingual_fixed.cfg \
    --output output_model_immo_ner_bilingual_v3 \
    --paths.train train_bilingual_V3.spacy \
    --paths.dev dev_bilingual_V3.spacy

# 3. Tester
python 3_test_model.py
```

---

## 📞 Support

- **Issues:** [GitHub Issues](https://github.com/your-username/NLP-urbanova/issues)
- **Documentation spaCy:** https://spacy.io/usage
- **Email:** your.email@example.com

---

## 🎓 Ressources Additionnelles

- [Documentation spaCy NER](https://spacy.io/usage/training#ner)
- [Guide de Prompt Engineering](https://docs.anthropic.com/claude/docs/prompt-engineering)
- [Exemples de Patterns REGEX](https://regex101.com/)

---

## ✅ Checklist de Démarrage

- [ ] Python 3.7+ installé
- [ ] Repository cloné
- [ ] Environnement virtuel créé
- [ ] Dépendances installées (`pip install -r requirements.txt`)
- [ ] Modèle téléchargé ou entraîné
- [ ] Test simple réussi
- [ ] Script de test exécuté

---

**Vous êtes prêt! 🚀**