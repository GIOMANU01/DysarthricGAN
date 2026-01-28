import os
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import toml
from cleese_stim.engines.phase_vocoder.phase_vocoder import PhaseVocoder

init_folder = "C:/Users/gioel/Desktop/patient_GAN/patient_F05"

# Percorsi
input_folder = os.path.join(init_folder, "trimmed_CF05")
check_folder = os.path.join(init_folder, "trimmed_F05")
output_folder = os.path.join(init_folder, "cleese_CF05")
excel_file = os.path.join(init_folder, "risultati_MFA.xlsx")
config_file = "C:/Users/gioel/Desktop/preprocessing/cleese-phase-vocoder.toml"

os.makedirs(output_folder, exist_ok=True)

# Leggi Excel
df = pd.read_excel(excel_file)
report = []

# Carica configurazione TOML
config = toml.load(config_file)
config["main"]["outPath"] = output_folder
config["main"]["param_ext"] = ".txt"

# Loop su tutti i file 
for i, row in df.iterrows():
    name = str(row["FILE NAME"]).strip()
    diff = float(row["differenza"])
    
    file_dis = os.path.join(check_folder, f"{name}.wav")
    file_sano = os.path.join(input_folder, f"{name}.wav")

    if not (os.path.exists(file_dis) and os.path.exists(file_sano)):
        print(f"File mancante: {name}")
        continue

    # Carica audio
    dis, sr = librosa.load(file_dis, sr=22050)
    sano, sr2 = librosa.load(file_sano, sr=22050)

    dur_dis = librosa.get_duration(y=dis, sr=sr)
    dur_sano = librosa.get_duration(y=sano, sr=sr)

    # Durata target 
    target_duration = dur_sano + diff
    if target_duration <= 0:
        print(f"Durata target <= 0 per {name}, salto")
        continue

    # Fattore di stretching
    s = target_duration / dur_sano  # >1 = allunga, <1 = velocizza

    # Applica CLEESE PhaseVocoder 
    # Imposta il BPF manualmente come array [[0, s]] per stretching statico
    BPF = np.array([[0.0, s]])
    sano_mod, _ = PhaseVocoder.process(
        soundData=sano,
        config=config,
        BPF=BPF,
        sample_rate=sr,
        sample_format='float32',
        file_output=False
    )

    # Allinea esattamente al disartrico 
    len_target = len(dis)
    sano_mod = sano_mod[:len_target]
    if len(sano_mod) < len_target:
        sano_mod = np.pad(sano_mod, (0, len_target - len(sano_mod)), mode='constant')

    dur_mod = librosa.get_duration(y=sano_mod, sr=sr)
    residuo = dur_mod - dur_dis

    # Salva 
    out_path = os.path.join(output_folder, f"{name}.wav")
    sf.write(out_path, sano_mod, sr)

    report.append({
        "FILE NAME": name,
        "Durata sano": dur_sano,
        "Durata modificata": dur_mod,
        "Durata disartrico": dur_dis,
        "Residuo finale": residuo
    })

# Report finale
report_file = os.path.join(os.path.dirname(output_folder), "report_durate_cleese.xlsx")
pd.DataFrame(report).to_excel(report_file, index=False)
print("\n Report salvato in:", report_file)
