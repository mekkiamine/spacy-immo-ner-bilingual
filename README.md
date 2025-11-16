# NLP-Urbanova - Modèle NER Immobilier Bilingue

Modèle de reconnaissance d'entités nommées (NER) pour les annonces immobilières en français et anglais.

## 🎯 Objectif

Extraire automatiquement les entités clés des annonces immobilières:
- **BEDS**: Nombre de chambres
- **BATHS**: Nombre de salles de bain
- **AREA**: Surface (m², sq ft)
- **PRICE**: Prix et devises
- **LOCATION**: Villes et quartiers
- **TYPE**: Type de propriété (appartement, villa, etc.)
- **TRANSACTION**: Type de transaction (vente, location)
- **AMENITY**: Équipements (piscine, jardin)
- **GARAGE**: Parking/garage
- **CONDITION**: État (neuf, rénové)

## 📊 Performance

- **Taux de détection**: 92-95%
- **Performance**: <50ms par requête
- **Dataset**: 3907 exemples d'entraînement
- **Langues**: Français & Anglais

## 🚀 Installation
```bash
# Cloner le repository
git clone https://github.com/votre-username/NLP-urbanova.git
cd NLP-urbanova

# Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Sur Windows: .venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

## 📝 Usage

### 1. Générer les Annotations
```bash
python 1_annotate_data.py
```

### 2. Préparer les Données
```bash
python 2_train_model.py
```

### 3. Entraîner le Modèle
```bash
python -m spacy train config_bilingual_fixed.cfg \
    --output output_model_immo_ner_bilingual_v3 \
    --paths.train train_bilingual_V3.spacy \
    --paths.dev dev_bilingual_V3.spacy
```

### 4. Tester le Modèle
```bash
python 3_test_model.py
```

## 📁 Structure du Projet
```
NLP-urbanova/
├── 1_annotate_data.py        # Génération des annotations
├── 2_train_model.py           # Préparation des données
├── 3_test_model.py            # Tests du modèle
├── config_bilingual_fixed.cfg # Configuration spaCy
├── house_price_bd.csv         # Dataset d'entraînement
└── requirements.txt           # Dépendances Python
```

## 🎓 Entités Reconnues

| Entité | Description | Exemples |
|--------|-------------|----------|
| BEDS | Chambres | "3 bedrooms", "4 chambres" |
| BATHS | Salles de bain | "2 bathrooms", "1 salle de bain" |
| AREA | Surface | "120 m²", "1800 sq ft" |
| PRICE | Prix | "450000 TND", "$50,000" |
| LOCATION | Localisation | "Tunis", "Dhaka", "La Marsa" |
| TYPE | Type de bien | "villa", "appartement", "S+3" |
| TRANSACTION | Transaction | "vendre", "louer", "sale" |
| AMENITY | Équipements | "piscine", "pool", "jardin" |
| GARAGE | Parking | "garage", "parking 2 places" |
| CONDITION | État | "neuf", "moderne", "rénové" |

## 🔧 Technologies

- **spaCy 3.7+**: Framework NER
- **Python 3.7+**
- **pandas**: Manipulation de données
- **jsonlines**: Format de données

## 📈 Exemples
```python
import spacy

# Charger le modèle
nlp = spacy.load("output_model_immo_ner_bilingual_v3/model-best")

# Analyser un texte
text = "Appartement S+3 de 120 m² à LOUER à Tunis. Prix: 800 TND/mois"
doc = nlp(text)

# Afficher les entités
for ent in doc.ents:
    print(f"{ent.text} → {ent.label_}")

# Output:
# Appartement → TYPE
# S+3 → TYPE
# 120 m² → AREA
# LOUER → TRANSACTION
# Tunis → LOCATION
# 800 TND → PRICE
```

## 📄 Licence

MIT License

## 👤 Auteur

Mohamed Amine Mekki- [@mekkiamine](https://github.com/mekkiamine)
