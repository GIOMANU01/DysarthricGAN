import os

# Percorso della cartella contenente i file
cartella = "C:/Users/gioel/Desktop/patient_GAN_vad/patient_F02/F02"  

# Ciclo su tutti i file nella cartella
for filename in os.listdir(cartella):
    if filename.endswith(".wav") and "_segment1" in filename:
        nuovo_nome = filename.replace("_segment1", "")
        vecchio_percorso = os.path.join(cartella, filename)
        nuovo_percorso = os.path.join(cartella, nuovo_nome)
        os.rename(vecchio_percorso, nuovo_percorso)
        print(f"Rinominato: {filename} -> {nuovo_nome}")
