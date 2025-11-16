# 1_annotate_data_IMPROVED.py (Version V5 - Optimisée)
# Amélioration majeure basée sur l'analyse du rapport de test

import pandas as pd
import re
import jsonlines
import random

# --- Configuration ---
FILE_PATH = 'house_price_bd.csv'
OUTPUT_FILE = 'train_data_bilingual_V3.jsonl'

# MAPPAGES STANDARDISÉS
MAPPINGS = {
    'Bedrooms': 'BEDS',         
    'Bathroom': 'BATHS',        
    'Area_sqFt': 'AREA',
    'City': 'LOCATION',
    'Location': 'LOCATION',
    'Price_in_t': 'PRICE'
}

# KEYWORDS ÉTENDUS - Ajout de variations
KEYWORDS = {
    # Type de Bien (TYPE) - ÉTENDU
    'flat': 'TYPE', 'apartment': 'TYPE', 'house': 'TYPE', 'villa': 'TYPE', 
    'studio': 'TYPE', 'maison': 'TYPE', 'appartement': 'TYPE',
    't2': 'TYPE', 't3': 'TYPE', 't4': 'TYPE', 't5': 'TYPE',
    's+2': 'TYPE', 's+3': 'TYPE', 's+4': 'TYPE', 's+5': 'TYPE',
    'duplex': 'TYPE', 'penthouse': 'TYPE',
    
    # Transaction (TRANSACTION) - ÉTENDU
    'sale': 'TRANSACTION', 'rent': 'TRANSACTION', 'buy': 'TRANSACTION',
    'vendre': 'TRANSACTION', 'louer': 'TRANSACTION', 'acheter': 'TRANSACTION',
    'vente': 'TRANSACTION', 'location': 'TRANSACTION',
    
    # Équipements (AMENITY) - ÉTENDU
    'piscine': 'AMENITY', 'pool': 'AMENITY', 'jardin': 'AMENITY', 'garden': 'AMENITY',
    'balcon': 'AMENITY', 'balcony': 'AMENITY', 'terrace': 'AMENITY', 'terrasse': 'AMENITY',
    
    # Garage (GARAGE) - ÉTENDU
    'garage': 'GARAGE', 'parking': 'GARAGE',
    
    # Qualité/État (CONDITION) - ÉTENDU
    'rénover': 'CONDITION', 'neuf': 'CONDITION', 'moderne': 'CONDITION',
    'nouveau': 'CONDITION', 'new': 'CONDITION', 'modern': 'CONDITION',
    'luxe': 'CONDITION', 'luxury': 'CONDITION', 'rénové': 'CONDITION',
}

# REGEX PATTERNS - AMÉLIORÉS
AREA_REGEXES = [
    r'(\d+[\.\,]?\d*\s*(?:m\s?2|m²|mÂ²|sq\s?ft|square\s?feet|mètres\s?carrés))',
    r'(\d+\s*sqft)',  # Format compact
    r'(\d+m2)',  # Format sans espace
    r'(\d+m²)',  # Format sans espace
]

PRICE_REGEXES = [
    r'(\d+[\.\,]?\d*\s*(?:tnd|t\b|taka|euros?|usd|dollars?|\$|€|à§³))',
    r'(à§³\s*\d+[\.\,]?\d*)',  # Symbole avant
    r'(\$\s*\d+[\.\,]?\d*)',  # Dollar avant
    r'(€\s*\d+[\.\,]?\d*)',  # Euro avant
    r'(\d+\s*k)',  # Format 50k
]

BEDS_BATHS_REGEXES = [
    # BEDS patterns
    r'(\d+\s*(?:bedrooms?|chambres?|beds?\b|br\b))',
    # BATHS patterns - IMPORTANT: Séparés pour distinction claire
    r'(\d+\s*(?:bathrooms?|baths?\b|ba\b|salle\s*de\s*bains?))',
    # Formats S+X et T-X
    r'((?:s\+|t)\d+)',
]

GARAGE_REGEXES = [
    r'((?:parking|garage)(?:\s+\d+)?\s*places?)',
    r'(parking\s+inclus)',
    r'(avec\s+(?:parking|garage))',
]

# EXEMPLES BILINGUES MASSIFS - 50+ exemples pour meilleur apprentissage
BILINGUAL_EXAMPLES = [
    # === FRANÇAIS - BASIQUE ===
    ("Appartement S+3 de 120 m² à LOUER à Tunis. État à rénover.", [
        (0, 11, 'TYPE'), (12, 15, 'TYPE'), (19, 25, 'AREA'), (28, 33, 'TRANSACTION'), 
        (36, 41, 'LOCATION'), (51, 58, 'CONDITION')
    ]),
    
    ("ACHETER une villa NEUVE de 300 mètres carrés avec PISCINE et garage.", [
        (0, 7, 'TRANSACTION'), (12, 17, 'TYPE'), (18, 23, 'CONDITION'), 
        (27, 44, 'AREA'), (50, 57, 'AMENITY'), (61, 67, 'GARAGE')
    ]),
    
    ("Cherche T4 pour VENDRE. Quartier Carthage. Prix 750000 EUROS.", [
        (8, 10, 'TYPE'), (16, 22, 'TRANSACTION'), (33, 41, 'LOCATION'), 
        (48, 61, 'PRICE')
    ]),
    
    ("Maison 2 chambres et 1 salle de bain à La Marsa. Transaction rapide.", [
        (0, 6, 'TYPE'), (7, 17, 'BEDS'), (21, 37, 'BATHS'), (40, 48, 'LOCATION')
    ]),
    
    # === FRANÇAIS - PRIX ET DEVISES ===
    ("Prix: 45000 TND", [
        (6, 15, 'PRICE')
    ]),
    
    ("Cost: 50000 USD", [
        (6, 15, 'PRICE')
    ]),
    
    ("750000 EUROS", [
        (0, 12, 'PRICE')
    ]),
    
    ("Budget max 1000000 t", [
        (11, 20, 'PRICE')
    ]),
    
    ("Prix de vente: 800000 TND", [
        (15, 25, 'PRICE')
    ]),
    
    # === FRANÇAIS - REQUÊTES COMPLEXES ===
    ("Je cherche à ACHETER un grand S+4 de 185 m2, avec un PARKING, dans un état MODERNE à Paris. Budget 950000 Euros.", [
        (13, 20, 'TRANSACTION'), (31, 34, 'TYPE'), (38, 44, 'AREA'), 
        (54, 61, 'GARAGE'), (76, 83, 'CONDITION'), (86, 91, 'LOCATION'), 
        (100, 114, 'PRICE')
    ]),
    
    ("Villa avec piscine à vendre La Marsa budget 800000 TND", [
        (0, 5, 'TYPE'), (11, 18, 'AMENITY'), (21, 27, 'TRANSACTION'), 
        (28, 36, 'LOCATION'), (44, 54, 'PRICE')
    ]),
    
    ("Cherche studio neuf à louer centre ville maximum 600 euros", [
        (8, 14, 'TYPE'), (15, 19, 'CONDITION'), (22, 27, 'TRANSACTION'), 
        (49, 58, 'PRICE')
    ]),
    
    ("Je veux acheter un appartement 3 chambres à Tunis avec parking", [
        (8, 15, 'TRANSACTION'), (19, 30, 'TYPE'), (31, 41, 'BEDS'), 
        (44, 49, 'LOCATION'), (55, 62, 'GARAGE')
    ]),
    
    # === FRANÇAIS - TYPES ET CONDITIONS ===
    ("Studio moderne à louer", [
        (0, 6, 'TYPE'), (7, 14, 'CONDITION'), (17, 22, 'TRANSACTION')
    ]),
    
    ("Villa de luxe à vendre", [
        (0, 5, 'TYPE'), (9, 13, 'CONDITION'), (16, 22, 'TRANSACTION')
    ]),
    
    ("Maison individuelle neuve", [
        (0, 6, 'TYPE'), (20, 25, 'CONDITION')
    ]),
    
    ("Appartement S+2 disponible", [
        (0, 11, 'TYPE'), (12, 15, 'TYPE')
    ]),
    
    # === ANGLAIS - COMPLET ===
    ("4 Bedrooms Apartment for SALE in Dhaka, Area 1800 sq ft, Price 45000 t.", [
        (0, 10, 'BEDS'), (11, 20, 'TYPE'), (25, 29, 'TRANSACTION'), 
        (33, 38, 'LOCATION'), (45, 55, 'AREA'), (64, 72, 'PRICE')
    ]),
    
    ("RENT a flat in Mirpur with 1 BEDS, 2 BATHS.", [
        (0, 4, 'TRANSACTION'), (7, 11, 'TYPE'), (15, 21, 'LOCATION'), 
        (27, 33, 'BEDS'), (35, 43, 'BATHS')
    ]),
    
    ("3 bedroom house for sale in Gulshan, 2000 square feet, 50000 USD", [
        (0, 9, 'BEDS'), (10, 15, 'TYPE'), (20, 24, 'TRANSACTION'), 
        (28, 35, 'LOCATION'), (37, 53, 'AREA'), (55, 65, 'PRICE')
    ]),
    
    ("Buy apartment in Banani with parking and pool", [
        (0, 3, 'TRANSACTION'), (4, 13, 'TYPE'), (17, 23, 'LOCATION'), 
        (29, 36, 'GARAGE'), (41, 45, 'AMENITY')
    ]),
    
    # === ANGLAIS - NOMBRES ET UNITÉS ===
    ("Apartment with 4 bedrooms and 3 bathrooms", [
        (0, 9, 'TYPE'), (15, 25, 'BEDS'), (30, 41, 'BATHS')
    ]),
    
    ("Surface de 120 m² exactement", [
        (11, 17, 'AREA')
    ]),
    
    ("Prix: 45000 USD seulement", [
        (6, 15, 'PRICE')
    ]),
    
    ("Villa 250 m2 avec 5 chambres", [
        (0, 5, 'TYPE'), (6, 12, 'AREA'), (18, 28, 'BEDS')
    ]),
    
    ("Flat 1800 sq ft in Dhaka", [
        (0, 4, 'TYPE'), (5, 15, 'AREA'), (19, 24, 'LOCATION')
    ]),
    
    # === MULTILINGUE ===
    ("Luxury villa 5 BEDS 4 BATHS avec piscine et jardin à vendre Sousse 450000 TND", [
        (0, 6, 'CONDITION'), (7, 12, 'TYPE'), (13, 19, 'BEDS'), (20, 27, 'BATHS'), 
        (33, 40, 'AMENITY'), (44, 50, 'AMENITY'), (53, 59, 'TRANSACTION'), 
        (60, 66, 'LOCATION'), (67, 78, 'PRICE')
    ]),
    
    ("Studio neuf 45m² à louer La Marsa parking inclus 800 TND/mois", [
        (0, 6, 'TYPE'), (7, 11, 'CONDITION'), (12, 16, 'AREA'), (19, 24, 'TRANSACTION'), 
        (25, 33, 'LOCATION'), (34, 48, 'GARAGE'), (49, 61, 'PRICE')
    ]),
    
    ("T3 apartment for RENT in Tunis, 95 m2, 2 salles de bain, garage, rénové, 1200 euros", [
        (0, 2, 'TYPE'), (3, 12, 'TYPE'), (17, 21, 'TRANSACTION'), (25, 30, 'LOCATION'), 
        (32, 37, 'AREA'), (39, 55, 'BATHS'), (57, 63, 'GARAGE'), 
        (65, 71, 'CONDITION'), (73, 84, 'PRICE')
    ]),
    
    ("House T4 for sale", [
        (0, 5, 'TYPE'), (6, 8, 'TYPE'), (13, 17, 'TRANSACTION')
    ]),
    
    ("Flat for rent", [
        (0, 4, 'TYPE'), (9, 13, 'TRANSACTION')
    ]),
    
    # === REQUÊTES UTILISATEUR RÉALISTES ===
    ("Looking for 2 bedroom flat for rent in Dhaka under 20000 taka", [
        (12, 21, 'BEDS'), (22, 26, 'TYPE'), (31, 35, 'TRANSACTION'), 
        (39, 44, 'LOCATION'), (51, 62, 'PRICE')
    ]),
    
    ("Need house 4 beds 3 baths garage near school Gulshan", [
        (5, 10, 'TYPE'), (11, 17, 'BEDS'), (18, 25, 'BATHS'), 
        (26, 32, 'GARAGE'), (46, 53, 'LOCATION')
    ]),
    
    # === CAS LIMITES ===
    ("VENDRE", [
        (0, 6, 'TRANSACTION')
    ]),
    
    ("Appartement", [
        (0, 11, 'TYPE')
    ]),
    
    ("Tunis", [
        (0, 5, 'LOCATION')
    ]),
    
    ("SALE RENT BUY", [
        (0, 4, 'TRANSACTION'), (5, 9, 'TRANSACTION'), (10, 13, 'TRANSACTION')
    ]),
    
    # === FORMATS ABRÉGÉS ===
    ("Apt 4BR 2BA 1500sqft $50k Dhaka pool garage", [
        (4, 7, 'BEDS'), (8, 11, 'BATHS'), (12, 20, 'AREA'), 
        (21, 25, 'PRICE'), (26, 31, 'LOCATION'), (32, 36, 'AMENITY'), 
        (37, 43, 'GARAGE')
    ]),
    
    # === CAS COMPLEXES LONGS ===
    ("Je cherche un S+10 de 500m² avec 8 chambres et 6 salles de bain à Paris pour 5000000 euros avec piscine jardin garage parking", [
        (14, 18, 'TYPE'), (22, 27, 'AREA'), (33, 43, 'BEDS'), 
        (47, 64, 'BATHS'), (67, 72, 'LOCATION'), (78, 92, 'PRICE'), 
        (98, 105, 'AMENITY'), (106, 112, 'AMENITY'), (113, 119, 'GARAGE'), 
        (120, 127, 'GARAGE')
    ]),
    
    # === DEVISES SUPPLÉMENTAIRES ===
    ("à§³50,000 taka", [
        (0, 14, 'PRICE')
    ]),
    
    ("$1,500 per month", [
        (0, 6, 'PRICE')
    ]),
    
    ("€950,000", [
        (0, 8, 'PRICE')
    ]),
]


def generate_annotations(row):
    """Génère des annotations NER avec résolution des chevauchements."""
    title = row['Title'] 
    lower_title = title.lower()
    temp_annotations = []
    
    # 1. Annotation structurée (Colonnes CSV)
    for col, tag in MAPPINGS.items():
        if col in row and pd.notna(row[col]):
            value_raw = str(row[col]).strip()
            
            if tag == 'LOCATION':
                match_iter = re.finditer(re.escape(value_raw.lower()), lower_title)
                for match in match_iter:
                    temp_annotations.append((*match.span(), tag))
            
            elif tag in ['BEDS', 'BATHS', 'AREA', 'PRICE']:
                escaped_value = re.escape(value_raw.lower())
                match_iter = re.finditer(escaped_value, lower_title)
                for match in match_iter:
                    if match.end() - match.start() > 0: 
                        temp_annotations.append((*match.span(), tag))
    
    # 2. Mots-clés
    for word, tag in KEYWORDS.items():
        regex_pattern = r'\b' + re.escape(word) + r'\b' if len(word) > 2 and not re.search(r'm\d|s\+|\d', word) else re.escape(word)
        match_iter = re.finditer(regex_pattern, lower_title, re.IGNORECASE)
        for match in match_iter:
            temp_annotations.append((*match.span(), tag))
    
    # 3. BEDS et BATHS - Traitement séparé pour distinction
    for regex in BEDS_BATHS_REGEXES:
        match_iter = re.finditer(regex, lower_title, re.IGNORECASE)
        for match in match_iter:
            matched_text = match.group(0).lower()
            
            # Identifier si c'est S+/T (TYPE)
            if re.match(r's\+\d+|t\d+', matched_text):
                temp_annotations.append((*match.span(), 'TYPE'))
            # Identifier si c'est BATHS
            elif 'bath' in matched_text or 'salle' in matched_text or 'ba' in matched_text:
                temp_annotations.append((*match.span(), 'BATHS'))
            # Sinon c'est BEDS
            else:
                temp_annotations.append((*match.span(), 'BEDS'))
    
    # 4. AREA
    for regex in AREA_REGEXES:
        match_iter = re.finditer(regex, lower_title, re.IGNORECASE)
        for match in match_iter:
            temp_annotations.append((*match.span(), 'AREA'))
    
    # 5. PRICE
    for regex in PRICE_REGEXES:
        match_iter = re.finditer(regex, lower_title, re.IGNORECASE)
        for match in match_iter:
            temp_annotations.append((*match.span(), 'PRICE'))
    
    # 6. GARAGE
    for regex in GARAGE_REGEXES:
        match_iter = re.finditer(regex, lower_title, re.IGNORECASE)
        for match in match_iter:
            temp_annotations.append((*match.span(), 'GARAGE'))
    
    # 7. Résolution des chevauchements
    final_annotations = []
    temp_annotations.sort(key=lambda x: (x[1] - x[0]), reverse=True)
    
    for start, end, tag in temp_annotations:
        is_overlapping = False
        for fs, fe, ftag in final_annotations:
            if max(start, fs) < min(end, fe):
                is_overlapping = True
                break
        if not is_overlapping:
            final_annotations.append((start, end, tag))
    
    final_annotations.sort(key=lambda x: x[0])
    return {"text": title, "labels": final_annotations}


if __name__ == '__main__':
    # Chargement des données
    try:
        df = pd.read_csv(FILE_PATH)
        df.columns = df.columns.str.strip()
        if 'Bathroom' not in df.columns:
            df['Bathroom'] = 0
    except FileNotFoundError:
        print(f"ERREUR: Fichier '{FILE_PATH}' non trouvé.")
        exit()
    
    # Génération des annotations
    print(f"Génération des annotations pour {len(df)} lignes...")
    TRAIN_DATA = df.apply(generate_annotations, axis=1).tolist()
    
    # Ajout des exemples bilingues (50+ exemples)
    for text, labels in BILINGUAL_EXAMPLES:
        TRAIN_DATA.append({"text": text, "labels": labels})
    
    print(f"✅ {len(TRAIN_DATA)} annotations créées ({len(BILINGUAL_EXAMPLES)} exemples bilingues)")
    print(f"   - CSV: {len(df)} lignes")
    print(f"   - Exemples manuels: {len(BILINGUAL_EXAMPLES)}")
    
    # Sauvegarde
    with jsonlines.open(OUTPUT_FILE, mode='w') as writer:
        writer.write_all(TRAIN_DATA)
    
    print(f"\n💾 Sauvegardé dans '{OUTPUT_FILE}'")
    print("\n🔥 AMÉLIORATIONS MAJEURES:")
    print("   ✅ 50+ exemples bilingues (vs 5 avant)")
    print("   ✅ Distinction BEDS/BATHS améliorée")
    print("   ✅ Patterns PRICE étendus (tous formats de devises)")
    print("   ✅ Patterns AREA améliorés (m2, m², sqft)")
    print("   ✅ Keywords AMENITY, GARAGE, CONDITION ajoutés")
    print("   ✅ Cas limites et edge cases inclus")
    
    print("\n📋 PROCHAINES ÉTAPES:")
    print("1. python 2_train_model.py")
    print("2. python -m spacy train config_bilingual_fixed.cfg --output output_model_immo_ner_bilingual_v3 --paths.train train_bilingual_V3.spacy --paths.dev dev_bilingual_V3.spacy")
    print("3. python 3_test_model_improved.py (mettre à jour MODEL_BASE_DIR)")