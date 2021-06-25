import tkinter as tk

from program.bayesian_network import bayesianNetwork
from program.clustering import kMeansCluster
from program.data_manager import readCSV
from program.nutriscore_classification import knn_model

food_l = ['energy_100g', 'fat_100g', 'saturated-fat_100g', 'monounsaturated-fat_100g',
          'polyunsaturated-fat_100g', 'omega-3-fat_100g', 'carbohydrates_100g', 'sugars_100g', 'fiber_100g',
          'proteins_100g', 'salt_100g', 'sodium_100g', 'vitamin-a_100g', 'vitamin-d_100g', 'vitamin-e_100g',
          'vitamin-c_100g', 'vitamin-b1_100g', 'vitamin-b2_100g', 'vitamin-pp_100g', 'vitamin-b6_100g',
          'vitamin-b9_100g', 'vitamin-b12_100g', 'pantothenic-acid_100g', 'potassium_100g', 'calcium_100g',
          'phosphorus_100g', 'iron_100g', 'magnesium_100g', 'zinc_100g', 'iodine_100g']
food_a = ['energy_100g', 'fat_100g', 'saturated-fat_100g', 'carbohydrates_100g', 'sugars_100g', 'fiber_100g',
          'proteins_100g', 'salt_100g', 'nutriscore_score', 'nova_group', 'additives_n']
food_b = ['product_name', 'energy_100g', 'fat_100g', 'proteins_100g', 'carbohydrates_100g']



# feature_selection_recursive_elimination(food_df, 20, food_l)

class Dialogo(tk.Frame):
    def __init__(self):
        self.energy = 0
        self.prot = 0
        self.carb = 0
        self.gras = 0
        self.food_df = readCSV('../data/food_dataset_final.csv', ',')
        self.hypers_knn = dict(leaf_size=1, n_neighbors=7, p=1, weights='distance')

        tk.Frame.__init__(self)
        self.master.title("Food machine learning")  # Diamo il titolo alla finestra.
        self.master.minsize(300, 400)  # Dimensioni minime della finestra
        self.grid(sticky=tk.E + tk.W + tk.N + tk.S)

        # self.ris = tk.StringVar()  # Questa variabile stringa viene usata per
        # self.ris.set("---")  # aggiornare la gui quando il risultato cambia.

        # Rendiamo ridimensionabile la finestra dell'applicazione
        top = self.winfo_toplevel()
        top.rowconfigure(0, weight=1)
        top.columnconfigure(0, weight=1)

        for i in range(13):
            self.rowconfigure(i, weight=1)
            self.columnconfigure(1, weight=1)

        self.etichetta1 = tk.Label(self, text="Calorie (per 100g):")  # Etichetta delle calorie
        self.etichetta1.grid(column=0, row=0, sticky=tk.W, padx=5, pady=5)

        self.entrata1 = tk.Entry(self)  # Casella d'inserimento delle calorie
        self.entrata1.grid(column=1, row=0, sticky=tk.E, padx=5, pady=5)

        self.etichetta2 = tk.Label(self, text="Carboidrati (per 100g):")  # Etichetta dei carboidrati
        self.etichetta2.grid(column=0, row=1, sticky=tk.W, padx=5, pady=5)

        self.entrata2 = tk.Entry(self)  # Casella d'inserimento dei carboidrati
        self.entrata2.grid(column=1, row=1, sticky=tk.E, padx=5, pady=5)

        self.etichetta3 = tk.Label(self, text="Proteine (per 100g):")  # Etichetta delle proteine
        self.etichetta3.grid(column=0, row=2, sticky=tk.W, padx=5, pady=5)

        self.entrata3 = tk.Entry(self)  # Casella d'inserimento delle proteine
        self.entrata3.grid(column=1, row=2, sticky=tk.E, padx=5, pady=5)

        self.etichetta4 = tk.Label(self, text="Grassi (per 100g):")  # Etichetta dei grassi
        self.etichetta4.grid(column=0, row=3, sticky=tk.W, padx=5, pady=5)

        self.entrata4 = tk.Entry(self)  # Casella d'inserimento dei grassi
        self.entrata4.grid(column=1, row=3, sticky=tk.E, padx=5, pady=5)

        self.bottone1 = tk.Button(self, text="Analisi", command=self.calcola)  # Bottone "Analisi"
        self.bottone1.grid(column=1, row=4, sticky=tk.E, padx=5, pady=5)
        self.bottone2 = tk.Button(self, text="Reset", command=self.reset)  # Bottone "Reset"
        self.bottone2.grid(column=0, row=4, sticky=tk.E, padx=5, pady=5)

        self.risultato_knn = tk.Text(self, height=1, width=50)  # Testo che mostra il risultato del knn.
        self.risultato_knn.grid(column=2, row=5, sticky=tk.E, padx=5, pady=5)

        self.risultato_kmeans = tk.Text(self, height=20, width=50)  # Testo che mostra il risultato del k-means.
        self.risultato_kmeans.grid(column=2, row=7, sticky=tk.E, padx=5, pady=5)

        self.risultato_bayes = tk.Text(self, height=1, width=50)  # Testo che mostra il risultato del knn.
        self.risultato_bayes.grid(column=2, row=6, sticky=tk.E, padx=5, pady=5)

    # Raccogliamo l'input e calcoliamo
    def calcola(self):
        self.risultato_knn.delete("1.0", "end")
        self.risultato_kmeans.delete("1.0", "end")
        self.risultato_bayes.delete("1.0", "end")
        if len(self.entrata1.get()) != 0:
            self.energy = float(self.entrata1.get())
        else:
            self.energy = 0
        self.energy *= 4.184
        if len(self.entrata2.get()) != 0:
            self.carb = float(self.entrata2.get())
        else:
            self.carb = 0
        if len(self.entrata3.get()) != 0:
            self.prot = float(self.entrata3.get())
        else:
            self.prot = 0
        if len(self.entrata1.get()) != 0:
            self.gras = float(self.entrata4.get())
        else:
            self.gras = 0
        values = {'energy_100g': self.energy, 'fat_100g': self.gras, 'carbohydrates_100g': self.carb,
                  'proteins_100g': self.prot}
        result_knn = knn_model(self.food_df, food_l, self.hypers_knn, values)
        result_kmeans = kMeansCluster(self.food_df, food_b, values)
        result_bayes = bayesianNetwork(self.food_df, values)
        self.risultato_knn.insert(tk.END, "Nutriscore = " + result_knn)
        self.risultato_bayes.insert(tk.END, result_bayes)
        self.risultato_kmeans.insert(tk.END, result_kmeans)

    def reset(self):
        self.risultato_knn.delete("1.0", "end")
        self.risultato_kmeans.delete("1.0", "end")
        self.risultato_bayes.delete("1.0", "end")
        self.entrata1.delete(0, "end")
        self.entrata2.delete(0, "end")
        self.entrata3.delete(0, "end")
        self.entrata4.delete(0, "end")


# Avvio del programma a condizione che non sia caricato come modulo
if __name__ == "__main__":
    d = Dialogo()
    d.mainloop()
