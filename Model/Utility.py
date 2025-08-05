import os

def create_folder(filename):

    folder_path = f"Figure/{os.path.splitext(filename)[0]}"

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Dossier créé : {folder_path}")
    else:
        print(f"Le dossier existe déjà : {folder_path}\n")

    return folder_path


def myPrint(*args, **kwargs):
    if IS_PRINT:
        print(*args, **kwargs)


        
from mainOpti import IS_PRINT
