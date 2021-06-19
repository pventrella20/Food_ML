import tkinter as tk

from program.data_manager import readCSV
from program.nutriscore_classification import knn_model, knn_classifier

food_l = ['energy_100g', 'fat_100g', 'saturated-fat_100g', 'monounsaturated-fat_100g',
          'polyunsaturated-fat_100g', 'omega-3-fat_100g', 'carbohydrates_100g', 'sugars_100g', 'fiber_100g',
          'proteins_100g', 'salt_100g', 'sodium_100g', 'vitamin-a_100g', 'vitamin-d_100g', 'vitamin-e_100g',
          'vitamin-c_100g', 'vitamin-b1_100g', 'vitamin-b2_100g', 'vitamin-pp_100g', 'vitamin-b6_100g',
          'vitamin-b9_100g', 'vitamin-b12_100g', 'pantothenic-acid_100g', 'potassium_100g', 'calcium_100g',
          'phosphorus_100g', 'iron_100g', 'magnesium_100g', 'zinc_100g', 'iodine_100g']
food_b = ['energy_100g', 'fat_100g', 'proteins_100g', 'carbohydrates_100g', 'fiber_100g', 'salt_100g']


# print(food_df[food_b].describe())
#
# kMeansCategorization(food_df, food_l)
#
# column_analisys(food_df, 'cluster')

# correlation_matrix(food_df[food_l])

# pair_plot(food_df)

# box_plot(food_df[food_l])

# data_analisys(food_df, 'iodine_100g')

# column_analisys(food_df, 'pnns_groups_2')

# feature_selection_recursive_elimination(food_df, 20, food_l)



# knn_regressor(food_df, food_l, 26)

# rf_classifier(food_df, food_l, 26)

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

        self.ris = tk.StringVar()  # Questa variabile stringa viene usata per
        self.ris.set("---")  # aggiornare la gui quando il risultato cambia.
        # Rendiamo ridimensionabile la finestra dell'applicazione
        top = self.winfo_toplevel()
        top.rowconfigure(0, weight=1)
        top.columnconfigure(0, weight=1)

        for i in range(13): self.rowconfigure(i, weight=1)
        self.columnconfigure(1, weight=1)

        self.etichetta1 = tk.Label(self, text="Calorie (per 100g):")  # Etichetta del lato 1
        self.etichetta1.grid(column=0, row=0, sticky=tk.W, padx=5, pady=5)

        self.entrata1 = tk.Entry(self)  # Casella d'inserimento del lato 1
        self.entrata1.grid(column=1, row=0, sticky=tk.E, padx=5, pady=5)

        self.etichetta2 = tk.Label(self, text="Carboidrati (per 100g):")  # Etichetta del lato 2
        self.etichetta2.grid(column=0, row=1, sticky=tk.W, padx=5, pady=5)

        self.entrata2 = tk.Entry(self)  # Casella d'inserimento del lato 2
        self.entrata2.grid(column=1, row=1, sticky=tk.E, padx=5, pady=5)

        self.etichetta3 = tk.Label(self, text="Proteine (per 100g):")  # Etichetta del lato 3
        self.etichetta3.grid(column=0, row=2, sticky=tk.W, padx=5, pady=5)

        self.entrata3 = tk.Entry(self)  # Casella d'inserimento del lato 3
        self.entrata3.grid(column=1, row=2, sticky=tk.E, padx=5, pady=5)

        self.etichetta4 = tk.Label(self, text="Grassi (per 100g):")  # Etichetta del lato 3
        self.etichetta4.grid(column=0, row=3, sticky=tk.W, padx=5, pady=5)

        self.entrata4 = tk.Entry(self)  # Casella d'inserimento del lato 3
        self.entrata4.grid(column=1, row=3, sticky=tk.E, padx=5, pady=5)

        self.bottone1 = tk.Button(self, text="Analisi", command=self.calcola)  # Bottone "Calcola"
        self.bottone1.grid(column=1, row=4, sticky=tk.E, padx=5, pady=5)
        self.bottone2 = tk.Button(self, text="Reset", command=self.reset)  # Bottone "Calcola"
        self.bottone2.grid(column=0, row=4, sticky=tk.E, padx=5, pady=5)

        self.risultato = tk.Label(self, textvariable=self.ris)  # Testo che mostra il risultato.
        self.risultato.grid(column=1, row=5, sticky=tk.E, padx=5, pady=5)

    # Raccogliamo l'input e calcoliamo l'area
    def calcola(self):
        self.energy = float(self.entrata1.get())
        self.energy *= 4.184
        self.carb = float(self.entrata2.get())
        self.prot = float(self.entrata3.get())
        self.gras = float(self.entrata3.get())
        values = {'energy_100g': self.energy, 'fat_100g': self.gras, 'carbohydrates_100g': self.carb,
                  'proteins_100g': self.prot}
        result = knn_model(self.food_df, food_l, self.hypers_knn, values)
        self.ris.set("Nutriscore = " + result)

    def reset(self):
        self.ris.set("---")
        self.entrata1.delete(0,"end")
        self.entrata2.delete(0, "end")
        self.entrata3.delete(0, "end")
        self.entrata4.delete(0, "end")


# Avvio del programma a condizione che non sia caricato come modulo
if __name__ == "__main__":
    d = Dialogo()
    d.mainloop()
