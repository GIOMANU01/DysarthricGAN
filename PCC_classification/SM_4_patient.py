import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import pandas as pd
from sequence_matcher import sequence_matcher
from scipy import stats
import os
import numpy as np
import matplotlib.pyplot as plt
import gc

# CONFIGURAZIONE PERCORSI 
patient = 'M04_new_2'  # Paziente specifico
primary_folder = f'/home/deepfake/DysarthricGAN/patients_gen/{patient}'
excel_parole_path = '/home/deepfake/DysarthricGAN/patients/lista_parole.xlsx'
output_dir = '/home/deepfake/DysarthricGAN/patients_gen/results_M04_2'
plots_dir = os.path.join(output_dir, 'plots')
os.makedirs(output_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

# SETUP DISPOSITIVO E MODELLO 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_id = "openai/whisper-large-v3"

print(f'Using device: {device}')

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
).to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

# CARICAMENTO PAROLE DI RIFERIMENTO 
df_words = pd.read_excel(excel_parole_path)
WORDS = [x for _, x in sorted(zip(df_words.iloc[:,1], df_words.iloc[:,0]))]

# Label clinica per questo paziente
label = [2]

# FUNZIONE PER PROCESSARE AUDIO 
def process_audio(file_path):
    result = pipe(
        file_path, 
        generate_kwargs={
            "task": "transcribe",
            "language": "english",
            "max_new_tokens": 40
        }
    )
    return result['text'].lower().strip()

# PROCESSAMENTO AUDIO 
print(f"--- Inizio Processamento Audio per {patient} ---")
word_transcription = []
files = sorted([f for f in os.listdir(primary_folder) if f.endswith(('.wav', '.mp3'))])

for i, file in enumerate(files):
    path = os.path.join(primary_folder, file)
    try:
        print(f"  [{i+1}/{len(files)}] {file}", end="... ", flush=True)
        out = process_audio(path)
        word_transcription.append(out)
        print(f"OK: '{out}'")
    except Exception as e:
        print(f"FALLITO: {e}")
        word_transcription.append("")
    
    if i % 10 == 0:
        torch.cuda.empty_cache()
        gc.collect()

# CALCOLO INTELLIGIBILITÀ SM 
Ip_tot = []
for item, GT in zip(word_transcription, WORDS):
    try:
        Ip_tot.append(sequence_matcher(item, GT))
    except:
        Ip_tot.append(0.0)

mean_val = np.mean(Ip_tot)
print(f"  >> Media SM {patient}: {mean_val:.4f}")

# --- ANALISI DATI E EXCEL ---
results_df = pd.DataFrame({
    'Patient': [patient],
    'SM_Score': [mean_val],
    'Label': label
})

excel_output = os.path.join(output_dir, f'risultati_finali_{patient}.xlsx')
results_df.to_excel(excel_output, index=False)
print(f"\n[INFO] Risultati salvati in: {excel_output}")







