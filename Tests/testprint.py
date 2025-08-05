DEBUG = True  # Passe à False pour désactiver

def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

# Exemple d'utilisation
debug_print("Ceci est un message de debug")