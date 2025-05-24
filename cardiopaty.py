import tkinter as tk
from tkinter import ttk, messagebox
import sv_ttk
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 1. Cargar datos
url = "https://raw.githubusercontent.com/GabrielCastro1221/csv_dataScience/main/heart1.csv"
df = pd.read_csv(url)

features_info = {
    'age': "Edad",
    'sex': "Sexo (1=Hombre, 0=Mujer)",
    'cp': "Tipo de dolor de pecho (0-3)",
    'trestbps': "Presión arterial en reposo (mm Hg)",
    'chol': "Colesterol sérico (mg/dl)",
    'fbs': "Azúcar en sangre en ayunas > 120 mg/dl (1=Sí, 0=No)",
    'restecg': "Resultado del electrocardiograma (0-2)",
    'thalach': "Frecuencia cardíaca máxima alcanzada",
    'exang': "Angina inducida por ejercicio (1=Sí, 0=No)",
    'oldpeak': "Depresión del ST inducida por el ejercicio",
    'slope': "Pendiente del segmento ST (0-2)",
    'ca': "Número de vasos principales (0-3)",
    'thal': "Defecto talámico (1=normal, 2=fijo, 3=reversible)"
}

X = df[list(features_info.keys())]
y = df['target']

# 2. Entrenar modelo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

param_grid = {'knn__n_neighbors': list(range(1, 21))}
grid = GridSearchCV(pipeline, param_grid, cv=5)
grid.fit(X_train, y_train)

resultado = {1: 'SANO', 0: 'ENFERMO'}

# 3. Crear GUI con inputs más anchos
class HeartApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Predicción de Enfermedades Cardíacas")
        self.root.geometry("850x450") 
        sv_ttk.set_theme("dark")  

        frame = ttk.Frame(root, padding=(20, 20))
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text="Ingrese datos del paciente", font=("Arial", 14, "bold")).grid(row=0, columnspan=2, pady=10)

        input_frame = ttk.Frame(frame)
        input_frame.grid(row=1, columnspan=2, pady=10)

        self.entries = {}
        for i, feature in enumerate(features_info.keys()):
            row = i % 7  
            col = i // 7  
            ttk.Label(input_frame, text=feature, font=("Arial", 10)).grid(row=row, column=col*2, sticky="w", padx=5, pady=5)

            entry = ttk.Entry(input_frame, width=40) 
            entry.grid(row=row, column=col*2+1, padx=5, pady=5)
            entry.insert(0, features_info[feature])  
            entry.bind("<FocusIn>", lambda event, e=entry: self.clear_placeholder(e))  
            entry.bind("<FocusOut>", lambda event, e=entry, f=feature: self.restore_placeholder(e, f))  

            self.entries[feature] = entry

        ttk.Button(frame, text="Predecir", command=self.predict, style="Accent.TButton").grid(row=2, columnspan=2, pady=20)

    def clear_placeholder(self, entry):
        """Borra el placeholder cuando el usuario empieza a escribir."""
        if entry.get() in features_info.values():
            entry.delete(0, tk.END)

    def restore_placeholder(self, entry, feature):
        """Restaura el placeholder si el campo quedó vacío."""
        if not entry.get():
            entry.insert(0, features_info[feature])

    def predict(self):
        try:
            user_data = [float(self.entries[f].get()) if self.entries[f].get() not in features_info.values() else 0.0 for f in features_info.keys()]
            prediccion = grid.predict([user_data])[0]
            messagebox.showinfo("Resultado", f"El modelo predice que la persona está: {resultado[prediccion]}")
            self.clear_fields()
        except ValueError:
            messagebox.showerror("Error", "Por favor, ingresa valores numéricos válidos.")

    def clear_fields(self):
        """Limpia los campos después de la predicción y restaura los placeholders."""
        for feature, entry in self.entries.items():
            entry.delete(0, tk.END)
            entry.insert(0, features_info[feature])  

# 4. Ejecutar la aplicación
root = tk.Tk()
app = HeartApp(root)
root.mainloop()
