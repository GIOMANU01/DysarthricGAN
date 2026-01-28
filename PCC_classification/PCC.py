from scipy.stats import pearsonr

# I tuoi due vettori (es. predizioni del modello vs dati reali)
vettore_UA = [33.924, 17.1, 40.769, 82.818, 18.041, 2, 42.6, 28.502, 89.566, 75.341, 93, 32.501, 17.776, 56.541, 43.017]
#vettore_UA_+_GEN = [39.141, 7.617, 26.876, 75.603, 13.749, 2.0, 35.404, 21.048, 93.0, 73.757, 88.547, 37.439, 10.904, 55.221, 37.581]
vettore2 = [29, 6, 62, 95, 15, 2, 58, 28, 93, 86, 93, 62, 7, 90, 29]

corr, p_value = pearsonr(vettore_UA, vettore2)

print(f"Coefficiente di Pearson: {corr:.4f}")
print(f"P-value: {p_value:.6f}")
