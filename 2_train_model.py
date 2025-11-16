# 2_train_model_IMPROVED.py (Version V3 - Optimisée)

import spacy
from spacy.tokens import DocBin
from tqdm import tqdm
import jsonlines
import random

# --- Configuration ---
TRAIN_DATA_FILE = 'train_data_bilingual_V3.jsonl'
TRAIN_OUTPUT_FILE = 'train_bilingual_V3.spacy'
DEV_OUTPUT_FILE = 'dev_bilingual_V3.spacy'

def convert_data_to_docbin(json_file):
    """Charge les données JSONL et les convertit en DocBin."""
    nlp = spacy.blank("xx")
    doc_bin = DocBin()
    
    print(f"📦 Conversion de {json_file} vers DocBin...")
    
    skipped = 0
    total = 0
    
    try:
        with jsonlines.open(json_file) as reader:
            for item in tqdm(reader):
                total += 1
                text = item["text"]
                labels = item["labels"]
                
                doc = nlp.make_doc(text)
                ents = []
                
                for start, end, label in labels:
                    span = doc.char_span(start, end, label=label, alignment_mode="contract")
                    if span is not None:
                        ents.append(span)
                    else:
                        skipped += 1
                
                # Sauvegarder même si certaines entités ont échoué
                doc.ents = ents
                doc_bin.add(doc)
                
    except FileNotFoundError:
        print(f"❌ ERREUR: '{json_file}' non trouvé. Exécutez 1_annotate_data_IMPROVED.py d'abord.")
        exit()
    
    if skipped > 0:
        print(f"⚠️  {skipped} entités ignorées (problèmes d'alignement)")
    print(f"✅ {total} documents convertis")
    
    return doc_bin


if __name__ == '__main__':
    print("\n" + "="*70)
    print("  CONVERSION DES DONNÉES D'ENTRAÎNEMENT")
    print("="*70 + "\n")
    
    # 1. Conversion
    doc_bin = convert_data_to_docbin(TRAIN_DATA_FILE)
    
    # 2. Séparation train/dev avec ratio 80/20
    all_docs = list(doc_bin.get_docs(spacy.blank("xx").vocab))
    random.shuffle(all_docs)
    
    train_split = int(0.8 * len(all_docs))
    train_docs = all_docs[:train_split]
    dev_docs = all_docs[train_split:]
    
    train_doc_bin = DocBin(docs=train_docs)
    dev_doc_bin = DocBin(docs=dev_docs)
    
    # 3. Sauvegarde
    train_doc_bin.to_disk(TRAIN_OUTPUT_FILE)
    dev_doc_bin.to_disk(DEV_OUTPUT_FILE)
    
    print(f"\n✅ Fichiers créés:")
    print(f"   📄 {TRAIN_OUTPUT_FILE} : {len(train_docs)} documents (80%)")
    print(f"   📄 {DEV_OUTPUT_FILE} : {len(dev_docs)} documents (20%)")
    
    print("\n" + "="*70)
    print("  PROCHAINE ÉTAPE: ENTRAÎNEMENT")
    print("="*70)
    print("\nCommande à exécuter:")
    print(f"python -m spacy train config_bilingual_fixed.cfg \\")
    print(f"    --output output_model_immo_ner_bilingual_v3 \\")
    print(f"    --paths.train {TRAIN_OUTPUT_FILE} \\")
    print(f"    --paths.dev {DEV_OUTPUT_FILE}")
    
    print("\n💡 CONSEILS POUR L'ENTRAÎNEMENT:")
    print("   • Augmentez max_steps à 30000 si possible (meilleure convergence)")
    print("   • Surveillez le score ENTS_F - il devrait atteindre >95%")
    print("   • Le modèle se sauvegarde automatiquement au meilleur score")
    print("   • L'entraînement peut prendre 10-30 minutes sur CPU")
    print()