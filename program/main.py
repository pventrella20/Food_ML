import tkinter as tk
from tkinter import messagebox
from program.bayesian_network import bayesianNetwork
from program.clustering import kMeansCluster
from program.config import DATASET_PATH, FEATURES_CLUSTER, FEATURES_KNN, KNN_HYPERS
from program.data_manager import readCSV
from program.nutriscore_classification import knn_model
food_l = FEATURES_KNN
food_b = FEATURES_CLUSTER
class FoodDialog(tk.Frame):
    """Legacy Tkinter GUI. Prefer the web interface (`app.py`)."""
    def __init__(self):
        self.energy = 0
        self.prot = 0
        self.carb = 0
        self.fat = 0
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
        self.label_energy = tk.Label(self, text="Calories (per 100g):")
        self.label_energy.grid(column=0, row=0, sticky=tk.W, padx=5, pady=5)
        self.entry_energy = tk.Entry(self)
        self.entry_energy.grid(column=1, row=0, sticky=tk.E, padx=5, pady=5)
        self.label_carbs = tk.Label(self, text="Carbohydrates (per 100g):")
        self.label_carbs.grid(column=0, row=1, sticky=tk.W, padx=5, pady=5)
        self.entry_carbs = tk.Entry(self)
        self.entry_carbs.grid(column=1, row=1, sticky=tk.E, padx=5, pady=5)
        self.label_proteins = tk.Label(self, text="Proteins (per 100g):")
        self.label_proteins.grid(column=0, row=2, sticky=tk.W, padx=5, pady=5)
        self.entry_proteins = tk.Entry(self)
        self.entry_proteins.grid(column=1, row=2, sticky=tk.E, padx=5, pady=5)
        self.label_fats = tk.Label(self, text="Fats (per 100g):")
        self.label_fats.grid(column=0, row=3, sticky=tk.W, padx=5, pady=5)
        self.entry_fats = tk.Entry(self)
        self.entry_fats.grid(column=1, row=3, sticky=tk.E, padx=5, pady=5)
        tk.Label(self, text="<- NUTRISCORE (A-E)").grid(column=3, row=5, sticky=tk.W, padx=5, pady=5)
        tk.Label(self, text="<- IS IT HEALTHY?").grid(column=3, row=6, sticky=tk.W, padx=5, pady=5)
        tk.Label(self, text="<- SIMILAR FOODS").grid(column=3, row=7, sticky=tk.W, padx=5, pady=5)
        self.button_analyze = tk.Button(self, text="Analyze", command=self.calculate)
        self.button_analyze.grid(column=1, row=4, sticky=tk.E, padx=5, pady=5)
        self.button_reset = tk.Button(self, text="Reset", command=self.reset)
        self.button_reset.grid(column=0, row=4, sticky=tk.E, padx=5, pady=5)
        self.result_knn = tk.Text(self, height=1, width=50)
        self.result_knn.grid(column=2, row=5, sticky=tk.E, padx=5, pady=5)
        self.result_bayes = tk.Text(self, height=1, width=50)
        self.result_bayes.grid(column=2, row=6, sticky=tk.E, padx=5, pady=5)
        self.result_kmeans = tk.Text(self, height=20, width=50)
        self.result_kmeans.grid(column=2, row=7, sticky=tk.E, padx=5, pady=5)

    def calculate(self):
        error_m = 0
        exc = 1
        self.result_knn.delete("1.0", "end")
        self.result_kmeans.delete("1.0", "end")
        self.result_bayes.delete("1.0", "end")
        if len(self.entry_energy.get()) != 0:
            exc = self.validate_numeric_input("calories", self.entry_energy.get())
        else:
            error_m = 1
        if (exc == 0) and (error_m == 0):
            self.energy = float(self.entry_energy.get()) * 4.184
        if len(self.entry_carbs.get()) != 0:
            exc = self.validate_numeric_input("carbohydrates", self.entry_carbs.get())
        else:
            error_m = 1
        if (exc == 0) and (error_m == 0):
            self.carb = float(self.entry_carbs.get())
        if len(self.entry_fats.get()) != 0:
            exc = self.validate_numeric_input("fats", self.entry_fats.get())
        else:
            error_m = 1
        if (exc == 0) and (error_m == 0):
            self.fat = float(self.entry_fats.get())
        if len(self.entry_proteins.get()) != 0:
            exc = self.validate_numeric_input("proteins", self.entry_proteins.get())
        else:
            error_m = 1
        if (error_m == 1) and (exc == 0):
            messagebox.showerror("Error", "PLEASE FILL IN ALL VALUES")
        if (exc == 0) and (error_m == 0):
            self.prot = float(self.entry_proteins.get())
        values = {"energy_100g": self.energy, "fat_100g": self.fat,
                  "carbohydrates_100g": self.carb, "proteins_100g": self.prot}
        if (exc == 0) and (error_m == 0):
            result_knn = knn_model(self.food_df, food_l, self.hypers_knn, values)
            result_bayes = bayesianNetwork(self.food_df, values)
            result_kmeans = kMeansCluster(self.food_df, food_b, values)
            if len(result_kmeans) == 0:
                messagebox.showerror("TRY DIFFERENT VALUES",
                                     "The provided nutritional values did not produce any results")
            else:
                self.result_knn.insert(tk.END, "Nutri-Score = " + str(result_knn))
                self.result_kmeans.insert(tk.END, result_kmeans)
                self.result_bayes.insert(tk.END, result_bayes)

    def validate_numeric_input(self, feature, numberControl):
        ret = 0
        err_m = 0
        limit = 100 if feature in ("fats", "carbohydrates", "proteins") else 1000
        try:
            val = float(numberControl)
            if val > limit or val < 0:
                self.reset()
                err_m = 1
                ret = 1
        except ValueError:
            self.reset()
            err_m = 2
            ret = 1
        if err_m == 1:
            messagebox.showerror("ERROR",
                                 "Invalid value. CALORIES(0-1000) PROTEINS-FATS-CARBOHYDRATES(0-100)")
        elif err_m == 2:
            messagebox.showerror("ERROR", "Non-numeric values are not allowed. Please enter numbers")
        return ret

    def reset(self):
        self.result_knn.delete("1.0", "end")
        self.result_kmeans.delete("1.0", "end")
        self.result_bayes.delete("1.0", "end")
        self.entry_energy.delete(0, "end")
        self.entry_carbs.delete(0, "end")
        self.entry_proteins.delete(0, "end")
        self.entry_fats.delete(0, "end")
if __name__ == "__main__":
    dialog = FoodDialog()
    dialog.mainloop()


# Backward-compatible aliases
Dialogo = FoodDialog
