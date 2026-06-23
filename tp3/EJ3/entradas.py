import numpy as np
from PIL import Image
import os

PATH = os.path.dirname(os.path.abspath(__file__))
PATH_IMG = os.path.join(PATH, "sprites", "img")

print(PATH_IMG)


def escalar_0_1(arreglo):
    return arreglo / 255.0

def mantener_0_255(arreglo):
    return arreglo

def crear_entrada_sprite(path_sprite):

    imagen = Image.open(path_sprite)
    imagen = imagen.convert('L') # Convertir a escala de grises (por si las dudas)
    # float32 para poder normalizar con decimales
    arreglo_base = np.asarray(imagen, dtype=np.float32)
    arreglo_normalizado = mantener_0_255(arreglo_base)
    return arreglo_normalizado


POKEDEX = [
    {"id": 1, "name": "Bulbasaur"},
    {"id": 4, "name": "Charmander"},
    {"id": 7, "name": "Squirtle"},
    {"id": 10, "name": "Caterpie"},
    {"id": 11, "name": "Metapod"},
    {"id": 13, "name": "Weedle"},
    {"id": 14, "name": "Kakuna"},
    {"id": 25, "name": "Pikachu"},
    {"id": 27, "name": "Sandshrew"},
    {"id": 29, "name": "Nidoran♀"},
    {"id": 32, "name": "Nidoran♂"},
    {"id": 35, "name": "Clefairy"},
    {"id": 39, "name": "Jigglypuff"},
    {"id": 41, "name": "Zubat"},
    {"id": 43, "name": "Oddish"},
    {"id": 46, "name": "Paras"},
    {"id": 48, "name": "Venonat"},
    {"id": 50, "name": "Diglett"},
    {"id": 52, "name": "Meowth"},
    {"id": 60, "name": "Poliwag"},
    {"id": 63, "name": "Abra"},
    {"id": 69, "name": "Bellsprout"},
    {"id": 74, "name": "Geodude"},
    {"id": 81, "name": "Magnemite"},
    {"id": 100, "name": "Voltorb"},
    {"id": 116, "name": "Horsea"},
    {"id": 132, "name": "Ditto"},
    {"id": 140, "name": "Omanyte"},
    {"id": 151, "name": "Mew"},
]

NOMBRES = [pokemon["name"] for pokemon in POKEDEX]

ENTRADAS = [crear_entrada_sprite(os.path.join(PATH_IMG, f"{pokemon['id']}.png")) for pokemon in POKEDEX]
