import tkinter as tk
from tkinter import messagebox
from program.bayesian_network import bayesianNetwork
from program.clustering import kMeansCluster
from program.config import DATASET_PATH, FEATURES_CLUSTER, FEATURES_KNN, KNN_HYPERS
from program.data_manager import readCSV
from program.nutriscore_classification import knn_model
food_l = FEATURES_KNN
food_b = FEATURES_CLUSTER
class Dialogo(tk.Frame):
    """GUI Tkinter legacy. Preferire l'interfaccia web (app.py)."""
    def __init__(self):
        self.energy = 0
        self.prot = 0
        self.carb = 0
        self.gras = 0
        self.food_df = readCSV(DATASET_PATH, ",")
        self.hypers_knn = dict(KNN_HYPERS)
        tk.Frame.__init__(self)
        self.master.title("Food machine learning")
        self.master.minsize(300, 400)
        self.grid(sticky=tk.E + tk.W + tk.N + tk.S)
        top = self.winfo_toplevel()
        top.rowconfigure(0, weight=1)
        top.columnconfigure(0, weight=1)
        for i in range(13):
            self.rowconfigure(i, weight=1)
            self.columnconfigure(1, weight=1)
        self.etichetta1 = tk.Label(self, text="Calorie (per 100g):")
        self.etichetta1.grid(column=0, row=0, sticky=tk.W, padx=5, pady=5)
        self.entrata1 = tk.Entry(self)
        self.entrata1.grid(column=1, row=0, sticky=tk.E, padx=5, pady=5)
        self.etichetta2 = tk.Label(self, text="Carboidrati (per 100g):")
        self.etichetta2.grid(column=0, row=1, sticky=tk.W, padx=5, pady=5)
        self.entrata2 = tk.Entry(self)
        self.entrata2.grid(column=1, row=1, sticky=tk.E, padx=5, pady=5)
        self.etichetta3 = tk.Label(self, text="Proteine (per 100g):")
        self.etichetta3.grid(column=0, row=2, sticky=tk.W, padx=5, pady=5)
        self.entrata3 = tk.Entry(self)
        self.entrata3.grid(column=1, row=2, sticky=tk.E, padx=5, pady=5)
        self.etichetta4 = tk.Label(self, text="Grassi (per 100g):")
        self.etichetta4.grid(column=0, row=3, sticky=tk.W, padx=5, pady=5)
        self.entrata4 = tk.Entry(self)
        self.entrata4.grid(column=1, row=3, sticky=tk.E, padx=5, pady=5)
        tk.Label(self, text="<- NUTRISCORE (A-E)").grid(column=3, row=5, sticky=tk.W, padx=5, pady=5)
        tk.Label(self, text="<- E SALUTARE?").grid(column=3, row=6, sticky=tk.W, padx=5, pady=5)
        tk.Label(self, text="<- ALIMENTI SIMILI").grid(column=3, row=7, sticky=tk.W, padx=5, pady=5)
        self.bottone1 = tk.Button(self, text="Analisi", command=self.calcola)
        self.bottone1.grid(column=1, row=4, sticky=tk.E, padx=5, pady=5)
        self.bottone2 = tk.Button(self, text="Reset", command=self.reset)
        self.bottone2.grid(column=0, row=4, sticky=tk.E, padx=5, pady=5)
        self.risultato_knn = tk.Text(self, height=1, width=50)
        self.risultato_knn.grid(column=2, row=5, sticky=tk.E, padx=5, pady=5)
        self.risultato_bayes = tk.Text(self, height=1, width=50)
        self.risultato_bayes.grid(column=2, row=6, sticky=tk.E, padx=5, pady=5)
        self.risultato_kmeans = tk.Text(self, height=20, width=50)
        self.risultato_kmeans.grid(column=2, row=7, sticky=tk.E, padx=5, pady=5)
    def calcola(self):
        errorMex = 0
        exc = 1
        self.risultato_knn.delete("1.0", "end")
        self.risultato_kmeans.delete("1.0", "end")
        self.risultato_bayes.delete("1.0", "end")
        if len(self.entrata1.get()) != 0:
            exc = self.checkString("calorie", self.entrata1.get())
        else:
            errorMex = 1
        if (exc == 0) and (errorMex == 0):
            self.energy = float(self.entrata1.get()) * 4.184
        if len(self.entrata2.get()) != 0:
            exc = self.checkString("carboidrati", self.entrata2.get())
        else:
            errorMex = 1
        if (exc == 0) and (errorMex == 0):
            self.carb = float(self.entrata2.get())
        if len(self.entrata4.get()) != 0:
            exc = self.checkString("grassi", self.entrata4.get())
        else:
            errorMex = 1
        if (exc == 0) and (errorMex == 0):
            self.gras = float(self.entrata4.get())
        if len(self.entrata3.get()) != 0:
            exc = self.checkString("proteine", self.entrata3.get())
        else:
            errorMex = 1
        if (errorMex == 1) and (exc == 0):
            messagebox.showerror("Error", "INSERIRE TUTTI I VALORI")
        if (exc == 0) and (errorMex == 0):
            self.prot = float(self.entrata3.get())
        values = {"energy_100g": self.energy, "fat_100g": self.gras,
                  "carbohydrates_100g": self.carb, "proteins_100g": self.prot}
        if (exc == 0) and (errorMex == 0):
            result_knn = knn_model(self.food_df, food_l, self.hypers_knn, values)
            result_bayes = bayesianNetwork(self.food_df, values)
            result_kmeans = kMeansCluster(self.food_df, food_b, values)
            if len(result_kmeans) == 0:
                messagebox.showerror("REINSERIRE VALORI",
                                     "I VALORI NUTRIZIONALI INSERITI NON HANNO PRODOTTO RISULTATI")
            else:
                self.risultato_knn.insert(tk.END, "Nutriscore = " + str(result_knn))
                self.risultato_kmeans.insert(tk.END, result_kmeans)
                self.risultato_bayes.insert(tk.END, result_bayes)
    def checkString(self, feature, numberControl):
        ret = 0
        errMex = 0
        limit = 100 if feature in ("grassi", "carboidrati", "proteine") else 1000
        try:
            val = float(numberControl)
            if val > limit or val < 0:
                self.reset()
                errMex = 1
                ret = 1
        except ValueError:
            self.reset()
            errMex = 2
            ret = 1
        if errMex == 1:
            messagebox.showerror("ERROR",
                                 "Valore non corretto. CALORIE(0-1000) PROTEINE-GRASSI-CARBOIDRATI(0-100)")
        elif errMex == 2:
            messagebox.showerror("ERROR", "Valori non numerici non consentiti. Reinserire valori")
        return ret
    def reset(self):
        self.risultato_knn.delete("1.0", "end")
        self.risultato_kmeans.delete("1.0", "end")
        self.risultato_bayes.delete("1.0", "end")
        self.entrata1.delete(0, "end")
        self.entrata2.delete(0, "end")
        self.entrata3.delete(0, "end")
        self.entrata4.delete(0, "end")
if __name__ == "__main__":
    d = Dialogo()
    d.mainloop()
