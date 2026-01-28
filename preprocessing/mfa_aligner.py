import os
import pandas as pd
from pydub import AudioSegment
from textgrids import TextGrid  
import re
from difflib import SequenceMatcher

folder_init = "C:/Users/gioel/Desktop/patient_GAN/patient_M04"
folder_dis = os.path.join(folder_init, "M04")
folder_sano = os.path.join(folder_init, "CM04")
output_dis = os.path.join(folder_init, "trimmed_M04")
output_sano = os.path.join(folder_init, "trimmed_CM04")
aligned_dis = os.path.join(folder_init, "aligned_M04")
aligned_sano = os.path.join(folder_init, "aligned_CM04")
excel_path = os.path.join("C:/Users/gioel/Desktop", "codici_dysartria", "creazione coppie", "lista_parole.xlsx")
results_excel = os.path.join(folder_init, "risultati_MFA.xlsx")
failed_excel_path = os.path.join(folder_init, "file_non_creati.xlsx")


os.makedirs(output_dis, exist_ok=True)
os.makedirs(output_sano, exist_ok=True)
os.makedirs(aligned_dis, exist_ok=True)
os.makedirs(aligned_sano, exist_ok=True)

# Modello da usare
acoustic_model = "english_mfa"
dictionary_model = "english_mfa"


df = pd.read_excel(excel_path)

#Crea file .lab per MFA
for _, row in df.iterrows():
    word = str(row['WORD'])
    w_code = str(row['FILE NAME'])
    for base in [folder_dis, folder_sano]:
        path_lab = os.path.join(base, f"{w_code}.lab")
        with open(path_lab, "w", encoding="utf-8") as f:
            f.write(word)

#Allinea con MFA
print("Allineamento parlato sano...")
os.system(f'mfa align "{folder_sano}" {acoustic_model} {dictionary_model} "{aligned_sano}" --skip-quality-check')

print("Allineamento parlato disartrico...")
os.system(f'mfa align "{folder_dis}" {acoustic_model} {dictionary_model} "{aligned_dis}" --skip-quality-check')


results = []
failed_rows = []  # lista di dizionari: FILE NAME, WORD, SET, REASON
missing_words = [] # lista parole che mfa non riconosce o non ha nell dizionario



def normalize_word(w):
    #Rimuove caratteri non alfabetici e converte in minuscolo
    return re.sub(r"[^a-z]", "", w.lower())

def similarity(a, b):
    #Calcola la similarità tra due stringhe (0–1)
    return SequenceMatcher(None, a, b).ratio()

def process_alignment(wav_folder, aligned_folder, trimmed_folder, tipo):
    os.makedirs(trimmed_folder, exist_ok=True)

    for _, row in df.iterrows():
        codice = str(row['FILE NAME']).strip()
        parola = str(row['WORD']).strip()
        tg_path = os.path.join(aligned_folder, f"{codice}.TextGrid")
        wav_path = os.path.join(wav_folder, f"{codice}.wav")
        out_path = os.path.join(trimmed_folder, f"{codice}.wav")

        # controlli preliminari
        if not os.path.exists(tg_path):
            failed_rows.append({"FILE NAME": codice, "WORD": parola, "SET": tipo, "REASON": "TextGrid mancante"})
            continue
        if not os.path.exists(wav_path):
            failed_rows.append({"FILE NAME": codice, "WORD": parola, "SET": tipo, "REASON": "WAV mancante"})
            continue

        # leggi TextGrid 
        try:
            tg = TextGrid(tg_path)
        except Exception as e:
            failed_rows.append({"FILE NAME": codice, "WORD": parola, "SET": tipo, "REASON": f"Errore lettura TextGrid: {e}"})
            continue

        # trova tier 'word' 
        try:
            tier_names = list(tg.keys())
        except Exception:
            try:
                tier_names = [t.name for t in tg.tiers]
            except Exception:
                tier_names = []

        words_tier_name = next((tn for tn in tier_names if "word" in tn.lower()), None)
        if words_tier_name is None:
            failed_rows.append({"FILE NAME": codice, "WORD": parola, "SET": tipo, "REASON": "Nessun tier 'word'"})
            continue

        words_tier = tg[words_tier_name]

        # cerca parola 
        trovato = False
        parola_norm = normalize_word(parola)
        best_match = ("", 0.0)
        best_interval = None

        for interval in words_tier:
            text = (getattr(interval, "text", "") or "").strip()
            text_norm = normalize_word(text)
            if not text_norm:
                continue

            if text_norm == parola_norm:
                trovato = True
                best_interval = interval
                break

            sim = similarity(text_norm, parola_norm)
            if sim > best_match[1]:
                best_match = (text, sim)
                best_interval = interval

        # se trovato o match più simile 
        if best_interval:
            start = float(best_interval.xmin)
            end = float(best_interval.xmax)
            start_ms = max(0, int(round(start * 1000)))
            end_ms = int(round(end * 1000))

            try:
                audio = AudioSegment.from_wav(wav_path)
            except Exception as e:
                failed_rows.append({"FILE NAME": codice, "WORD": parola, "SET": tipo, "REASON": f"Impossibile aprire WAV: {e}"})
                continue

            audio_len = len(audio)
            if end_ms > audio_len:
                end_ms = audio_len
            if end_ms <= start_ms:
                failed_rows.append({"FILE NAME": codice, "WORD": parola, "SET": tipo, "REASON": "Durata non valida (end <= start)"})
                continue

            try:
                trimmed = audio[start_ms:end_ms]
                trimmed.export(out_path, format="wav")
            except Exception as e:
                failed_rows.append({"FILE NAME": codice, "WORD": parola, "SET": tipo, "REASON": f"Errore trim/export: {e}"})
                continue

            duration_sec = (end_ms - start_ms) / 1000.0
            results.append({
                "FILE NAME": codice,
                "WORD": parola,
                f"inizio_{tipo}": start,
                f"fine_{tipo}": end,
                f"durata_{tipo}": duration_sec
            })

            if not trovato and best_match[1] > 0.6:
                reason = f"Parola non trovata esatta. Match più simile: '{best_match[0]}' ({best_match[1]:.2f})"
                failed_rows.append({"FILE NAME": codice, "WORD": parola, "SET": tipo, "REASON": reason})
        else:
            failed_rows.append({"FILE NAME": codice, "WORD": parola, "SET": tipo, "REASON": "Parola non trovata nel TextGrid"})

# esecuzione
process_alignment(folder_dis, aligned_dis, output_dis, "disartrico")
process_alignment(folder_sano, aligned_sano, output_sano, "sano")

# salva risultati
df_results = pd.DataFrame(results)
if not df_results.empty:
    df_final = df_results.pivot_table(index=["FILE NAME", "WORD"], aggfunc="first").reset_index()
    if "durata_disartrico" in df_final.columns and "durata_sano" in df_final.columns:
        df_final["differenza"] = df_final["durata_disartrico"] - df_final["durata_sano"]
    else:
        df_final["differenza"] = pd.NA
    df_final.to_excel(results_excel, index=False)

# salva parole non trovate o approssimate 
if failed_rows:
    failed_df = pd.DataFrame(failed_rows).drop_duplicates()
    failed_df.to_excel(failed_excel_path, index=False)

print(f"WAV tagliati salvati in '{output_dis}' e '{output_sano}'")
print(f"Risultati Excel: {results_excel}")
print(f"Parole non trovate o approssimate: {failed_excel_path}")
