import json
import os
from pathlib import Path

folder_path = Path("./answers/google_ner/schreibmaschine_results")

if folder_path.exists() and folder_path.is_dir():
    all_files = list(folder_path.glob("*"))
    json_files = list(folder_path.glob("*.json"))
    print(f"Alle Dateien im Ordner: {len(all_files)}")
    print(f"JSON-Dateien gefunden: {len(json_files)}")
    if len(all_files) > 0 and len(json_files) == 0:
        print("Erste 5 Dateien im Ordner:")
        for f in all_files[:5]:
            print(f"  - {f.name}")
print("\n")

total_files = 0
denomination_not_null = 0
eco_not_null = 0
both_null = 0
both_not_null = 0
only_denomination = 0
only_eco = 0


for json_file in folder_path.glob("*.json"):
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if "content" in data and len(data["content"]) > 0:
            content = data["content"][0]
            denomination = content.get("denomination")
            eco = content.get("eco")
            
            total_files += 1
            
            has_denomination = denomination is not None
            has_eco = eco is not None
            
            if has_denomination:
                denomination_not_null += 1
            if has_eco:
                eco_not_null += 1
            
            if has_denomination and has_eco:
                both_not_null += 1
            elif has_denomination and not has_eco:
                only_denomination += 1
            elif not has_denomination and has_eco:
                only_eco += 1
            elif not has_denomination and not has_eco:
                both_null += 1
                
    except Exception as e:
        print(f"Fehler beim Verarbeiten von {json_file.name}: {e}")

print(f"\nGesamtanzahl verarbeiteter Dateien: {total_files}")
print("\n" + "-" * 60)
print("EINZELNE FELDER:")
print("-" * 60)
print(f"Dateien mit Konfession (nicht null): {denomination_not_null}")
print(f"Dateien mit Wirtschaft (nicht null): {eco_not_null}")
print("\n" + "-" * 60)
print("KOMBINATIONEN:")
print("-" * 60)
print(f"Beide Felder null:                     {both_null}")
print(f"Beide Felder nicht null:               {both_not_null}")
print(f"Nur Konfession nicht null:             {only_denomination}")
print(f"Nur Wirtschaft nicht null:             {only_eco}")
print("\n" + "=" * 60)

# Optional: Prozentuale Verteilung
if total_files > 0:
    print("\nPROZENTUALE VERTEILUNG:")
    print("-" * 60)
    print(f"Beide null:          {both_null/total_files*100:.1f}%")
    print(f"Beide nicht null:    {both_not_null/total_files*100:.1f}%")
    print(f"Konfession:          {only_denomination/total_files*100:.1f}%")
    print(f"Wirtschaft:          {only_eco/total_files*100:.1f}%")
    print("=" * 60)