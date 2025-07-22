# Skálafüggetlen gráfok vizualizációja – Streamlit alkalmazás

Ez a projekt különböző hálózati modellek (pl. Barabási–Albert, Watts–Strogatz, stb.) interaktív vizualizációját teszi lehetővé egy egyszerű webes felületen, a [Streamlit](https://streamlit.io/) keretrendszer segítségével.

## Főbb jellemzők

- Többféle, klasszikus gráfmodellek generálása és vizualizálása
- Paraméterek interaktív beállítása oldalsávon keresztül
- Részletes magyarázat minden modellhez

## Telepítés és futtatás lokálisan

1. **Követelmények telepítése** (virtuális környezet ajánlott):

    ```bash
    pip install -r requirements.txt
    ```

2. **Alkalmazás indítása:**

    ```bash
    streamlit run main.py
    ```

Ezután böngészőben megnyílik a webapp (alapesetben [http://localhost:8501](http://localhost:8501) címen).

## Deploy Streamlit Community Cloud-ra (streamlit.io)

1. Készíts egy GitHub repository-t, töltsd fel ide a `main.py` és a `requirements.txt` fájlokat (és ezt a readme-t).
2. Lépj be a [Streamlit Cloud](https://streamlit.io/cloud) felületére, majd válaszd ki a GitHub repót és a főfájlt (`main.py`).
3. Indítsd el az appot – néhány másodperc múlva elérhető lesz egy publikus linken.

## Fájlok

- `main.py` – maga a Streamlit app
- `requirements.txt` – szükséges Python csomagok listája

## Használt főbb technológiák

- **Python 3.8+**
- [Streamlit](https://streamlit.io/)
- [NetworkX](https://networkx.org/)
- [Matplotlib](https://matplotlib.org/)


