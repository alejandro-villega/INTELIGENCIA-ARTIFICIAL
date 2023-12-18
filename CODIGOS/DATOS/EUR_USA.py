# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 18:16:49 2023

@author: AlejandroVillega
"""

import requests

# URL de una API ficticia para obtener la cotización EUR/USD
api_url = "https://api.example.com/forex/EUR/USD"

try:
    # Hacer una solicitud GET a la API
    response = requests.get(api_url)

    # Verificar el estado de la respuesta
    if response.status_code == 200:
        # Obtener los datos de la respuesta (por ejemplo, en formato JSON)
        data = response.json()

        # Procesar los datos según sea necesario
        exchange_rate = data["exchange_rate"]

        # Alimentar los datos a la red neuronal o realizar otras operaciones
        print(f"Cotización EUR/USD: {exchange_rate}")
    else:
        print(f"Error en la solicitud: {response.status_code}")
        print(f"Contenido de la respuesta: {response.text}")
except Exception as e:
    print(f"Error al obtener los datos: {str(e)}")