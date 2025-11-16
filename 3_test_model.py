# 3_test_model_improved.py (Version Améliorée - V5)
# Tests complets avec métriques, cas limites, et analyse détaillée

import spacy
from spacy import displacy
from pathlib import Path
from collections import defaultdict, Counter
import json
from datetime import datetime

# --- Configuration ---
MODEL_BASE_DIR = 'output_model_immo_ner_bilingual_v3'
MODEL_PATH = Path(MODEL_BASE_DIR) / 'model-best'
OUTPUT_DIR = Path('test_results')
OUTPUT_DIR.mkdir(exist_ok=True)

# Labels du modèle
ALL_LABELS = [
    "BEDS", "BATHS", "AREA", "PRICE", "LOCATION", 
    "TYPE", "TRANSACTION", "AMENITY", "GARAGE", "CONDITION"
]

# Couleurs pour la visualisation (améliorées pour meilleur contraste)
COLORS = {
    "BEDS": "#FF6B6B",      # Rouge clair
    "BATHS": "#4ECDC4",     # Turquoise
    "AREA": "#45B7D1",      # Bleu ciel
    "PRICE": "#FFA07A",     # Saumon
    "LOCATION": "#98D8C8",  # Vert menthe
    "TYPE": "#F7DC6F",      # Jaune doré
    "TRANSACTION": "#BB8FCE", # Violet clair
    "AMENITY": "#85C1E2",   # Bleu pâle
    "GARAGE": "#ABEBC6",    # Vert pâle
    "CONDITION": "#F8B4D9"  # Rose
}
OPTIONS = {"colors": COLORS}


class NERModelTester:
    """Classe pour tester et évaluer un modèle NER personnalisé."""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.nlp = None
        self.test_results = {
            'timestamp': datetime.now().isoformat(),
            'model_path': str(model_path),
            'test_cases': [],
            'statistics': defaultdict(int),
            'entity_counts': defaultdict(int)
        }
    
    def load_model(self):
        """Charge le modèle NER avec gestion d'erreurs."""
        try:
            self.nlp = spacy.load(self.model_path)
            print(f"✅ Modèle chargé avec succès: {self.model_path}")
            print(f"   Pipeline: {self.nlp.pipe_names}")
            print(f"   Labels NER: {self.nlp.get_pipe('ner').labels}")
            return True
        except OSError:
            # Essayer model-last si model-best n'existe pas
            model_path_last = Path(MODEL_BASE_DIR) / 'model-last'
            try:
                self.nlp = spacy.load(model_path_last)
                print(f"✅ Modèle chargé (fallback): {model_path_last}")
                return True
            except OSError:
                print(f"❌ ERREUR: Impossible de charger le modèle.")
                print(f"   Vérifiez: {MODEL_BASE_DIR}/model-best ou model-last")
                return False
    
    def get_test_cases(self):
        """Retourne un ensemble complet de cas de test organisés par catégorie."""
        return {
            "Anglais - Basique": [
                "4 Bedrooms Apartment for SALE in Dhaka, Area 1800 sq ft, Price 45000 t.",
                "RENT a flat in Mirpur with 1 BEDS, 2 BATHS.",
                "3 bedroom house for sale in Gulshan, 2000 square feet, 50000 USD",
                "Buy apartment in Banani with parking and pool"
            ],
            
            "Français - Basique": [
                "Appartement S+3 de 120 m² à LOUER à Tunis. État à rénover.",
                "ACHETER une villa NEUVE de 300 mètres carrés avec PISCINE et garage.",
                "Cherche T4 pour VENDRE. Quartier Carthage. Prix 750000 EUROS.",
                "Maison 2 chambres et 1 salle de bain à La Marsa. Transaction rapide."
            ],
            
            "Cas Complexes - Multilingues": [
                "Je cherche à ACHETER un grand S+4 de 185 m2, avec un PARKING, dans un état MODERNE à Paris. Budget 950000 Euros.",
                "Luxury villa 5 BEDS 4 BATHS avec piscine et jardin à vendre Sousse 450000 TND",
                "Studio neuf 45m² à louer La Marsa parking inclus 800 TND/mois",
                "T3 apartment for RENT in Tunis, 95 m2, 2 salles de bain, garage, rénové, 1200 euros"
            ],
            
            "Nombres et Unités": [
                "Apartment with 4 bedrooms and 3 bathrooms",
                "Surface de 120 m² exactement",
                "Prix: 45000 USD seulement",
                "Villa 250 m2 avec 5 chambres",
                "Flat 1800 sq ft in Dhaka"
            ],
            
            "Types de Propriétés": [
                "Studio moderne à louer",
                "Villa de luxe à vendre",
                "Appartement S+2 disponible",
                "House T4 for sale",
                "Maison individuelle neuve",
                "Flat for rent"
            ],
            
            "Cas Limites et Edge Cases": [
                "VENDRE",  # Un seul mot
                "Appartement",  # Type seul
                "Tunis",  # Location seule
                "",  # Chaîne vide
                "123",  # Nombres seuls
                "SALE RENT BUY",  # Mots-clés multiples
                "appartement appartement appartement",  # Répétitions
                "Apt 4BR 2BA 1500sqft $50k Dhaka pool garage",  # Format abrégé
                "Je cherche un S+10 de 500m² avec 8 chambres et 6 salles de bain à Paris pour 5000000 euros avec piscine jardin garage parking",  # Très long
            ],
            
            "Devises et Prix": [
                "Prix: 45000 TND",
                "Cost: 50000 USD",
                "750000 EUROS",
                "à§³50,000 taka",
                "Budget max 1000000 t"
            ],
            
            "Requêtes Utilisateur Réalistes": [
                "Je veux acheter un appartement 3 chambres à Tunis avec parking",
                "Looking for 2 bedroom flat for rent in Dhaka under 20000 taka",
                "Villa avec piscine à vendre La Marsa budget 800000 TND",
                "Cherche studio neuf à louer centre ville maximum 600 euros",
                "Need house 4 beds 3 baths garage near school Gulshan"
            ]
        }
    
    def test_single_phrase(self, phrase, category=""):
        """Teste une phrase et retourne les résultats détaillés."""
        if not phrase.strip():
            return None
            
        doc = self.nlp(phrase)
        
        result = {
            'category': category,
            'text': phrase,
            'entities': [],
            'entity_count': len(doc.ents),
            'has_entities': len(doc.ents) > 0
        }
        
        for ent in doc.ents:
            entity_info = {
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            }
            result['entities'].append(entity_info)
            self.test_results['entity_counts'][ent.label_] += 1
        
        return result
    
    def run_comprehensive_tests(self):
        """Exécute tous les tests et compile les statistiques."""
        print("\n" + "="*80)
        print("🧪 TESTS COMPLETS DU MODÈLE NER - IMMOBILIER BILINGUE")
        print("="*80)
        
        test_cases = self.get_test_cases()
        all_results = []
        
        for category, phrases in test_cases.items():
            print(f"\n📋 Catégorie: {category}")
            print("-" * 80)
            
            for phrase in phrases:
                result = self.test_single_phrase(phrase, category)
                if result:
                    all_results.append(result)
                    self.test_results['test_cases'].append(result)
                    
                    # Affichage
                    print(f"\n   Texte: {phrase}")
                    if result['entities']:
                        for ent in result['entities']:
                            print(f"      ✓ '{ent['text']}' → {ent['label']}")
                    else:
                        print(f"      ⚠ Aucune entité détectée")
                    
                    self.test_results['statistics']['total_tests'] += 1
                    if result['has_entities']:
                        self.test_results['statistics']['tests_with_entities'] += 1
        
        # Calcul des statistiques finales
        self._calculate_statistics()
        self._print_statistics()
        
        return all_results
    
    def _calculate_statistics(self):
        """Calcule les statistiques globales."""
        stats = self.test_results['statistics']
        total = stats['total_tests']
        
        if total > 0:
            stats['detection_rate'] = (stats['tests_with_entities'] / total) * 100
            stats['avg_entities_per_test'] = sum(self.test_results['entity_counts'].values()) / total
    
    def _print_statistics(self):
        """Affiche un résumé des statistiques."""
        print("\n" + "="*80)
        print("📊 STATISTIQUES GLOBALES")
        print("="*80)
        
        stats = self.test_results['statistics']
        print(f"\n✅ Tests totaux: {stats['total_tests']}")
        print(f"✅ Tests avec entités détectées: {stats['tests_with_entities']}")
        print(f"✅ Taux de détection: {stats.get('detection_rate', 0):.2f}%")
        print(f"✅ Moyenne d'entités par test: {stats.get('avg_entities_per_test', 0):.2f}")
        
        print("\n📈 Répartition des Entités Détectées:")
        entity_counts = self.test_results['entity_counts']
        sorted_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)
        
        for label, count in sorted_entities:
            bar = "█" * (count // 5 + 1)
            print(f"   {label:12} : {count:3} {bar}")
    
    def generate_html_visualizations(self):
        """Génère des visualisations HTML pour différents cas de test."""
        print("\n" + "="*80)
        print("🎨 GÉNÉRATION DES VISUALISATIONS HTML")
        print("="*80)
        
        # Sélection des meilleures phrases pour la visualisation
        visualization_phrases = [
            ("Requête Complexe Française", 
             "Je cherche à ACHETER un grand S+4 de 185 m2, avec un PARKING, dans un état MODERNE à Paris. Budget 950000 Euros."),
            
            ("Requête Complexe Anglaise",
             "Looking for 4 Bedrooms house for SALE in Dhaka, 2000 sq ft, with pool and garage, price 75000 USD."),
            
            ("Requête Mixte Bilingue",
             "T3 apartment for RENT in Tunis, 95 m2, 2 salles de bain, garage, rénové, 1200 euros"),
            
            ("Cas Avec Tous Les Labels",
             "Villa NEUVE 5 chambres 4 baths 350m² PISCINE JARDIN GARAGE 2 places à VENDRE Carthage état MODERNE 980000 TND")
        ]
        
        html_files = []
        
        for title, phrase in visualization_phrases:
            doc = self.nlp(phrase)
            
            # Génération HTML
            html = displacy.render(doc, style="ent", options=OPTIONS, page=True)
            
            # Ajout d'un titre et de métadonnées
            html = html.replace(
                '<body>',
                f'<body><h1 style="text-align:center; color:#2C3E50;">{title}</h1>'
                f'<p style="text-align:center; color:#7F8C8D; font-style:italic;">"{phrase}"</p><hr>'
            )
            
            # Sauvegarde
            filename = OUTPUT_DIR / f"visualization_{title.lower().replace(' ', '_')}.html"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(html)
            
            html_files.append(filename)
            print(f"   ✓ {title}: {filename}")
        
        # Génération d'une page index
        self._generate_index_page(html_files)
        
        return html_files
    
    def _generate_index_page(self, html_files):
        """Génère une page index pour toutes les visualisations."""
        index_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>NER Model - Visualisations</title>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 40px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                }
                .container {
                    background: white;
                    border-radius: 10px;
                    padding: 40px;
                    box-shadow: 0 10px 40px rgba(0,0,0,0.1);
                }
                h1 {
                    color: #2C3E50;
                    text-align: center;
                    margin-bottom: 40px;
                }
                .card {
                    border: 2px solid #ecf0f1;
                    border-radius: 8px;
                    padding: 20px;
                    margin: 20px 0;
                    transition: transform 0.2s, box-shadow 0.2s;
                }
                .card:hover {
                    transform: translateY(-5px);
                    box-shadow: 0 5px 20px rgba(0,0,0,0.1);
                }
                .card a {
                    color: #3498db;
                    text-decoration: none;
                    font-size: 18px;
                    font-weight: bold;
                }
                .card a:hover {
                    color: #2980b9;
                }
                .legend {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px;
                    margin-top: 30px;
                }
                .legend-item {
                    display: flex;
                    align-items: center;
                    padding: 10px;
                    border-radius: 5px;
                    background: #f8f9fa;
                }
                .legend-color {
                    width: 30px;
                    height: 30px;
                    border-radius: 5px;
                    margin-right: 10px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🏠 NER Model - Visualisations Immobilières</h1>
                <p style="text-align: center; color: #7F8C8D;">
                    Modèle bilingue (Français/Anglais) pour l'extraction d'entités immobilières
                </p>
                
                <h2>📊 Visualisations Disponibles:</h2>
        """
        
        for html_file in html_files:
            title = html_file.stem.replace('visualization_', '').replace('_', ' ').title()
            index_html += f"""
                <div class="card">
                    <a href="{html_file.name}" target="_blank">
                        📄 {title}
                    </a>
                    <p style="color: #7F8C8D; margin: 10px 0 0 0;">
                        Cliquez pour voir la visualisation interactive
                    </p>
                </div>
            """
        
        # Ajout de la légende des couleurs
        index_html += """
                <h2>🎨 Légende des Entités:</h2>
                <div class="legend">
        """
        
        for label, color in COLORS.items():
            index_html += f"""
                <div class="legend-item">
                    <div class="legend-color" style="background-color: {color};"></div>
                    <strong>{label}</strong>
                </div>
            """
        
        index_html += """
                </div>
            </div>
        </body>
        </html>
        """
        
        index_path = OUTPUT_DIR / "index.html"
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(index_html)
        
        print(f"\n   ✓ Page d'index: {index_path}")
    
    def save_test_report(self):
        """Sauvegarde un rapport JSON complet des tests."""
        report_path = OUTPUT_DIR / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Rapport sauvegardé: {report_path}")
        return report_path
    
    def test_performance(self, num_iterations=100):
        """Teste les performances du modèle (vitesse de traitement)."""
        import time
        
        print("\n" + "="*80)
        print("⚡ TEST DE PERFORMANCES")
        print("="*80)
        
        test_phrase = "Appartement 3 chambres à louer Tunis 120m² 800 TND garage"
        
        # Warm-up
        for _ in range(10):
            _ = self.nlp(test_phrase)
        
        # Test réel
        start_time = time.time()
        for _ in range(num_iterations):
            doc = self.nlp(test_phrase)
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = (total_time / num_iterations) * 1000  # en millisecondes
        
        print(f"\n✅ Iterations: {num_iterations}")
        print(f"✅ Temps total: {total_time:.3f} secondes")
        print(f"✅ Temps moyen par requête: {avg_time:.2f} ms")
        print(f"✅ Requêtes par seconde: {num_iterations / total_time:.0f}")
        
        self.test_results['performance'] = {
            'iterations': num_iterations,
            'total_time': total_time,
            'avg_time_ms': avg_time,
            'requests_per_second': num_iterations / total_time
        }


def main():
    """Fonction principale pour exécuter tous les tests."""
    
    print("\n" + "🏠"*40)
    print(" "*30 + "TESTEUR NER - IMMOBILIER")
    print("🏠"*40 + "\n")
    
    # Initialisation
    tester = NERModelTester(MODEL_PATH)
    
    # Chargement du modèle
    if not tester.load_model():
        return
    
    # Tests complets
    tester.run_comprehensive_tests()
    
    # Test de performances
    tester.test_performance()
    
    # Génération des visualisations
    tester.generate_html_visualizations()
    
    # Sauvegarde du rapport
    tester.save_test_report()
    
    print("\n" + "="*80)
    print("✅ TOUS LES TESTS TERMINÉS AVEC SUCCÈS!")
    print("="*80)
    print(f"\n📁 Résultats disponibles dans: {OUTPUT_DIR}")
    print(f"🌐 Ouvrez '{OUTPUT_DIR}/index.html' pour voir toutes les visualisations")
    print("\n")


if __name__ == '__main__':
    main()