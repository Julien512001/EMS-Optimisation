import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def rmse_signals(x_fem, y_fem, x_model, y_model):

    xmin = max(np.min(x_fem), np.min(x_model))
    xmax = min(np.max(x_fem), np.max(x_model))

    mask_model = (x_model >= xmin) & (x_model <= xmax)
    x_model_common = x_model[mask_model]
    y_model_common = y_model[mask_model]

    y_fem_interp = np.interp(x_model_common, x_fem, y_fem)

    rmse = np.sqrt(np.mean((y_fem_interp - y_model_common) ** 2))
    return rmse

def rmse_signals_percent(x_fem, y_fem, x_model, y_model, ref='max'):

    rmse = rmse_signals(x_fem, y_fem, x_model, y_model)
    
    if ref == 'max':
        norm = np.max(np.abs(y_fem))
    elif ref == 'rms':
        norm = np.sqrt(np.mean(y_fem**2))
    else:
        raise ValueError("ref doit être 'max' ou 'rms'")
    
    return (rmse / norm) * 100


df = pd.read_csv('Validation/Compare/unsaturated/Bz_0.txt', 
                 sep='\t', 
                 comment='%')  # nom des colonnes
x_FEM = df.iloc[:,0].to_numpy()*1e-3
Bz_0_FEM = df.iloc[:,1].to_numpy()

df = pd.read_csv('Validation/Compare/unsaturated/Bx_0.txt', 
                 sep='\t', 
                 comment='%')  # nom des colonnes
Bx_0_FEM = df.iloc[:,1].to_numpy()



df = pd.read_csv('Validation/Compare/unsaturated/Bz_1.txt', 
                 sep='\t', 
                 comment='%')  # nom des colonnes
x_FEM = df.iloc[:,0].to_numpy()*1e-3
Bz_1_FEM = df.iloc[:,1].to_numpy() 

df = pd.read_csv('Validation/Compare/unsaturated/Bx_1.txt', 
                 sep='\t', 
                 comment='%')  # nom des colonnes
Bx_1_FEM = df.iloc[:,1].to_numpy()



df = pd.read_csv('Validation/Compare/unsaturated/Bz_2.txt', 
                 sep='\t', 
                 comment='%')  # nom des colonnes
x_FEM = df.iloc[:,0].to_numpy()*1e-3
Bz_2_FEM = df.iloc[:,1].to_numpy()

df = pd.read_csv('Validation/Compare/unsaturated/Bx_2.txt', 
                 sep='\t', 
                 comment='%')  # nom des colonnes
Bx_2_FEM = df.iloc[:,1].to_numpy()



df = pd.read_csv('Validation/Compare/unsaturated/Bz_3.txt', 
                 sep='\t', 
                 comment='%')  # nom des colonnes
x_FEM = df.iloc[:,0].to_numpy()*1e-3
Bz_3_FEM = df.iloc[:,1].to_numpy()

df = pd.read_csv('Validation/Compare/unsaturated/Bx_3.txt', 
                 sep='\t', 
                 comment='%')  # nom des colonnes
Bx_3_FEM = df.iloc[:,1].to_numpy()



df = pd.read_csv('Validation/Compare/unsaturated/Bz_4.txt', 
                 sep='\t', 
                 comment='%')  # nom des colonnes
x_FEM = df.iloc[:,0].to_numpy()*1e-3
Bz_4_FEM = df.iloc[:,1].to_numpy()

df = pd.read_csv('Validation/Compare/unsaturated/Bx_4.txt', 
                 sep='\t', 
                 comment='%')  # nom des colonnes
Bx_4_FEM = df.iloc[:,1].to_numpy()



df = pd.read_csv('Validation/Compare/unsaturated/champ_B_1_test.csv')

x = df['x'].to_numpy()
Bz_0 = df['Bz_0'].to_numpy()
Bx_0 = df["Bx_0"].to_numpy()
Bz_1 = df['Bz_1'].to_numpy()
Bx_1 = df["Bx_1"].to_numpy()
Bz_2 = df['Bz_2'].to_numpy()
Bx_2 = df["Bx_2"].to_numpy()
Bz_3 = df['Bz_3'].to_numpy()
Bx_3 = df["Bx_3"].to_numpy()
Bz_4 = df['Bz_4'].to_numpy()
Bx_4 = df["Bx_4"].to_numpy()



rmse_z0 = rmse_signals_percent(x_FEM, Bz_0_FEM, x, Bz_0)
rmse_x0 = rmse_signals_percent(x_FEM, Bx_0_FEM, x, Bx_0)
rmse_z1 = rmse_signals_percent(x_FEM, Bz_1_FEM, x, Bz_1)
rmse_x1 = rmse_signals_percent(x_FEM, Bx_1_FEM, x, Bx_1)
rmse_z2 = rmse_signals_percent(x_FEM, Bz_2_FEM, x, Bz_2)
rmse_x2 = rmse_signals_percent(x_FEM, Bx_2_FEM, x, Bx_2)
rmse_z3 = rmse_signals_percent(x_FEM, Bz_3_FEM, x, Bz_3)
rmse_x3 = rmse_signals_percent(x_FEM, Bx_3_FEM, x, Bx_3)
rmse_z4 = rmse_signals_percent(x_FEM, Bz_4_FEM, x, Bz_4)
rmse_x4 = rmse_signals_percent(x_FEM, Bx_4_FEM, x, Bx_4)
print(rmse_z0)
print(rmse_x0)
print(rmse_z1)
print(rmse_x1)
print(rmse_z2)
print(rmse_x2)
print(rmse_z3)
print(rmse_x3)
print(rmse_z4)
print(rmse_x4)


fig = False

# Palette plus sobre et adaptée au papier
couleur_Bx_FEM = "#8B0000"      # Rouge sombre
couleur_Bx_MEC = "#CD5C5C"      # Rouge clair/gris rosé
couleur_Bz_FEM = "#003366"      # Bleu foncé
couleur_Bz_MEC = "#7BAFD4"      # Bleu doux

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9
})

for i in range(5):
    # if (i != 3):
    #     continue
    plt.figure(figsize=(6, 4))
    plt.title(f"Comparaison des composantes $B_x$ et $B_z$ sur la coupe $H_{i}$")
    
    plt.plot(eval(f"x_FEM"), eval(f"Bx_{i}_FEM"), color=couleur_Bx_FEM, label=r"$B_x$ (FEM)", linewidth=1.5)
    plt.scatter(eval("x"), eval(f"Bx_{i}"), alpha=0.6, color=couleur_Bx_MEC, label=r"$B_x$ (MEC)", s=15)
    
    plt.plot(eval(f"x_FEM"), eval(f"Bz_{i}_FEM"), color=couleur_Bz_FEM, label=r"$B_z$ (FEM)", linewidth=1.5)
    plt.scatter(eval("x"), eval(f"Bz_{i}"), alpha=0.6, color=couleur_Bz_MEC, label=r"$B_z$ (MEC)", s=15)
    
    plt.xlabel("Position $x$ (m)")
    plt.ylabel("Champ magnétique $B$ (T)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="best")
    plt.tight_layout(pad=1.5)

    if fig:
        plt.savefig(f"Validation/Compare/unsaturated/valid1_L{i}.pdf", dpi=300)

plt.show()