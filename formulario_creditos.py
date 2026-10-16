"""Interfaz Tkinter para demostrar el clasificador KNN de riesgo crediticio.

Los parámetros de preprocesamiento se aprenden solo con el entrenamiento.
El formulario no incorpora solicitudes nuevas al conjunto de entrenamiento.
"""

import json
import logging
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from demostracion_creditos import COLUMNAS, clasificar_solicitud, entrenar_modelo

LOGGER = logging.getLogger(__name__)
RUTA_PROYECTO = Path(__file__).resolve().parent

# Nombres explicativos; los códigos categóricos mantienen el significado del dataset.
ETIQUETAS = {
    "laufkont": "Estado de cuenta corriente",
    "laufzeit": "Duración del crédito (meses)",
    "moral": "Historial crediticio",
    "verw": "Finalidad del crédito",
    "hoehe": "Monto del crédito",
    "sparkont": "Ahorros",
    "beszeit": "Antigüedad laboral",
    "rate": "Cuota / ingresos (categoría)",
    "famges": "Estado personal",
    "buerge": "Garantes",
    "wohnzeit": "Tiempo de residencia (categoría)",
    "verm": "Propiedades",
    "alter": "Edad (años)",
    "weitkred": "Otros créditos",
    "wohn": "Tipo de vivienda",
    "bishkred": "Créditos anteriores (categoría)",
    "beruf": "Ocupación",
    "pers": "Personas a cargo (categoría)",
    "telef": "Teléfono",
    "gastarb": "Trabajador extranjero",
}


# Los códigos corresponden a los valores del dataset South German Credit.
# Las categorías sin equivalencia verificada conservan una etiqueta neutra.
# No se cambian las codificaciones que recibe el modelo.
DESCRIPCIONES = {
    "verw": {
        0: "Otros fines", 1: "Automóvil nuevo", 2: "Automóvil usado",
        3: "Muebles o equipamiento", 4: "Radio o televisión",
        5: "Electrodomésticos", 6: "Reparaciones", 7: "Educación",
        8: "Vacaciones", 9: "Formación profesional", 10: "Negocio",
    },
    "beszeit": {
        1: "Sin empleo", 2: "Menos de 1 año", 3: "De 1 a menos de 4 años",
        4: "De 4 a menos de 7 años", 5: "7 años o más",
    },
    "wohn": {1: "Alquiler", 2: "Vivienda propia", 3: "Vivienda gratuita"},
    "telef": {1: "Sin teléfono registrado", 2: "Con teléfono registrado"},
    "gastarb": {1: "Sí", 2: "No"},
}

# Estas seis variables se ingresan como valores numéricos originales.
NUMERICAS = {"laufzeit", "hoehe", "rate", "wohnzeit", "alter", "bishkred"}


def opciones_categoricas(nombre, categorias):
    """Genera opciones descriptivas conservando los códigos del modelo."""
    codigos = sorted(int(valor) for valor in categorias)
    descripciones = CATEGORIAS.get(nombre, {})

    return {
        f"{descripciones.get(codigo, 'Categoría')} (código {codigo})": codigo
        for codigo in codigos
    }

# Descripciones de los códigos originales de South German Credit.
CATEGORIAS = {
    "laufkont": {
        1: "Sin cuenta corriente",
        2: "Saldo negativo",
        3: "Saldo entre 0 y menos de 200 DM",
        4: "Saldo de 200 DM o más / nómina",
    },
    "moral": {
        0: "Retrasos anteriores en los pagos",
        1: "Historial crítico u otros créditos",
        2: "Sin créditos o créditos pagados correctamente",
        3: "Créditos actuales pagados correctamente",
        4: "Créditos anteriores pagados correctamente en este banco",
    },
    "verw": {
        0: "Otros",
        1: "Automóvil nuevo",
        2: "Automóvil usado",
        3: "Muebles o equipamiento",
        4: "Radio o televisión",
        5: "Electrodomésticos",
        6: "Reparaciones",
        7: "Educación",
        8: "Vacaciones",
        9: "Capacitación profesional",
        10: "Negocio",
    },
    "sparkont": {
        1: "Sin ahorros o información desconocida",
        2: "Ahorros inferiores a 100 DM",
        3: "Ahorros entre 100 y menos de 500 DM",
        4: "Ahorros entre 500 y menos de 1000 DM",
        5: "Ahorros de 1000 DM o más",
    },
    "beszeit": {
        1: "Desempleado",
        2: "Menos de 1 año",
        3: "Entre 1 y menos de 4 años",
        4: "Entre 4 y menos de 7 años",
        5: "7 años o más",
    },
    "famges": {
        1: "Hombre divorciado o separado",
        2: "Mujer no soltera u hombre soltero",
        3: "Hombre casado o viudo",
        4: "Mujer soltera",
    },
    "buerge": {
        1: "Sin codeudor ni garante",
        2: "Con codeudor",
        3: "Con garante",
    },
    "verm": {
        1: "Sin propiedades o desconocidas",
        2: "Automóvil u otros bienes",
        3: "Seguro de vida o ahorro inmobiliario",
        4: "Bienes inmuebles",
    },
    "weitkred": {
        1: "Otros créditos bancarios",
        2: "Créditos en establecimientos comerciales",
        3: "Sin otros planes de crédito",
    },
    "wohn": {
        1: "Vivienda gratuita",
        2: "Vivienda alquilada",
        3: "Vivienda propia",
    },
    "beruf": {
        1: "Desempleado o no cualificado no residente",
        2: "Trabajador no cualificado residente",
        3: "Empleado cualificado",
        4: "Directivo, autónomo o altamente cualificado",
    },
    "pers": {
        1: "Tres o más personas dependientes",
        2: "De cero a dos personas dependientes",
    },
    "telef": {
        1: "Sin teléfono registrado",
        2: "Con teléfono registrado a su nombre",
    },
    "gastarb": {
        1: "Sí",
        2: "No",
    },
}

class FormularioCreditos(tk.Tk):
    """Ventana para capturar y evaluar solicitudes con un KNN reproducible."""

    def __init__(self):
        super().__init__()
        self.title("Riesgo crediticio | Demostración KNN")
        self.geometry("950x760")
        self.minsize(770, 650)
        self.configure(bg="#f5f7fb")
        self.modelo = None
        self.ejemplos = []
        self.campos = {}
        self.selectores = {}
        self.opciones = {}
        self._construir_interfaz()
        self.after(100, self._cargar_modelo)

    def _construir_interfaz(self):
        """Crea los controles, el área desplazable y la sección de resultados."""
        ttk.Style(self).theme_use("clam")
        encabezado = ttk.Frame(self, padding=(22, 18, 22, 12))
        encabezado.pack(fill="x")
        ttk.Label(
            encabezado, text="Evaluación de solicitudes de crédito",
            font=("Segoe UI", 19, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            encabezado,
            text="KNN vectorizado · K = 5 · 20 variables originales · 62 características",
            font=("Segoe UI", 10),
        ).pack(anchor="w", pady=(4, 0))

        opciones = ttk.Frame(self, padding=(22, 4, 22, 8))
        opciones.pack(fill="x")
        ttk.Button(
            opciones,
            text="Buen crédito",
            command=lambda: self._cargar_ejemplo(0),
        ).pack(side="left", padx=(0, 8))
        ttk.Button(
            opciones,
            text="Mal crédito",
            command=lambda: self._cargar_ejemplo(1),
        ).pack(side="left", padx=(0, 8))
        ttk.Button(opciones, text="Limpiar", command=self._limpiar).pack(side="left")

        contenedor = ttk.Frame(self, padding=(22, 0, 22, 0))
        contenedor.pack(fill="both", expand=True)
        lienzo = tk.Canvas(contenedor, highlightthickness=0, bg="#f5f7fb")
        barra = ttk.Scrollbar(contenedor, orient="vertical", command=lienzo.yview)
        lienzo.configure(yscrollcommand=barra.set)
        barra.pack(side="right", fill="y")
        lienzo.pack(side="left", fill="both", expand=True)
        formulario = ttk.Frame(lienzo, padding=12)
        ventana = lienzo.create_window((0, 0), window=formulario, anchor="nw")
        formulario.bind(
            "<Configure>",
            lambda _: lienzo.configure(scrollregion=lienzo.bbox("all")),
        )
        lienzo.bind(
            "<Configure>",
            lambda evento: lienzo.itemconfigure(ventana, width=evento.width),
        )
        formulario.columnconfigure(0, weight=1)
        formulario.columnconfigure(1, weight=1)

        for indice, nombre in enumerate(COLUMNAS):
            columna = indice % 2
            fila = indice // 2
            bloque = ttk.Frame(formulario, padding=(8, 6))
            bloque.grid(row=fila, column=columna, sticky="ew")
            bloque.columnconfigure(0, weight=1)
            ttk.Label(
                bloque, text=ETIQUETAS[nombre], font=("Segoe UI", 9),
            ).grid(row=0, column=0, sticky="w")
            variable = tk.StringVar()
            self.campos[nombre] = variable
            if nombre in NUMERICAS:
                control = ttk.Entry(bloque, textvariable=variable)
            else:
                control = ttk.Combobox(
                    bloque, textvariable=variable, state="readonly"
                )
                self.selectores[nombre] = control
            control.grid(row=1, column=0, sticky="ew", pady=(4, 0))

        pie = ttk.Frame(self, padding=(22, 12, 22, 18))
        pie.pack(fill="x")
        self.boton_evaluar = ttk.Button(
            pie, text="Evaluar crédito", command=self._evaluar, state="disabled"
        )
        self.boton_evaluar.pack(fill="x", ipady=7)
        self.resultado = tk.StringVar(value="Preparando el modelo de entrenamiento...")
        ttk.Label(
            pie, textvariable=self.resultado, font=("Segoe UI", 12, "bold"),
            wraplength=850, justify="left",
        ).pack(anchor="w", pady=(12, 4))

    def _cargar_modelo(self):
        """Reconstruye el KNN y recupera los ejemplos para la demostración."""
        try:
            self.modelo = entrenar_modelo()
            # Los códigos válidos se obtienen del conjunto de entrenamiento.
            # Así el formulario conserva la codificación exacta del KNN.
            from modelo.caracteristicas import COLUMNAS_CATEGORICAS

            for indice, categorias in zip(
                COLUMNAS_CATEGORICAS, self.modelo["categorias"]
            ):
                nombre = COLUMNAS[indice]
                opciones = opciones_categoricas(nombre, categorias)
                self.opciones[nombre] = opciones
                self.selectores[nombre]["values"] = tuple(opciones)
            ruta = RUTA_PROYECTO / "solicitudes_ejemplo.json"
            with ruta.open(encoding="utf-8") as archivo:
                self.ejemplos = json.load(archivo)
            if len(self.ejemplos) < 2:
                raise ValueError("Se requieren dos ejemplos en el archivo JSON")
            self.boton_evaluar.configure(state="normal")
            self.resultado.set("Modelo preparado. Introduce los datos.")
        except (OSError, ValueError, KeyError, TypeError) as error:
            LOGGER.exception("No fue posible preparar el modelo")
            self.resultado.set("Error al preparar el modelo.")
            messagebox.showerror("Error al iniciar", str(error))

    def _cargar_ejemplo(self, indice):
        """Muestra en el formulario un ejemplo sin modificar el entrenamiento."""
        if len(self.ejemplos) <= indice:
            messagebox.showwarning("Espera", "Todavía no se han cargado los ejemplos.")
            return
        ejemplo = self.ejemplos[indice]
        for nombre in COLUMNAS:
            valor = ejemplo.get(nombre)
            if nombre in self.opciones and valor is not None:
                codigo = int(valor)
                etiqueta = next(
                    (texto for texto, numero in self.opciones[nombre].items()
                     if numero == codigo),
                    "",
                )
                self.campos[nombre].set(etiqueta)
            else:
                self.campos[nombre].set("" if valor is None else str(valor))
        self.resultado.set("Ejemplo cargado. Pulsa 'Evaluar crédito'.")

    def _limpiar(self):
        """Borra únicamente los valores introducidos en el formulario."""
        for variable in self.campos.values():
            variable.set("")
        self.resultado.set("Introduce los datos de una nueva solicitud.")

    def _evaluar(self):
        """Valida los 20 valores y muestra la clasificación del KNN."""
        if self.modelo is None:
            messagebox.showwarning("Espera", "El modelo todavía no está preparado.")
            return
        try:
            solicitud = {}
            for nombre in COLUMNAS:
                texto = self.campos[nombre].get().strip()
                if not texto:
                    raise ValueError(f"Completa el campo: {ETIQUETAS[nombre]}")
                if nombre in self.opciones:
                    solicitud[nombre] = self.opciones[nombre][texto]
                else:
                    solicitud[nombre] = float(texto)
            clase = clasificar_solicitud(self.modelo, solicitud)
        except (ValueError, TypeError, KeyError) as error:
            messagebox.showerror("Datos no válidos", str(error))
            return
        if clase == 1:
            mensaje = "CLASE 1 — BUEN CRÉDITO (clasificación favorable)"
        else:
            mensaje = "CLASE 0 — MAL CRÉDITO (clasificación desfavorable)"
        self.resultado.set(mensaje + "\nPredicción educativa: requiere revisión humana.")
        LOGGER.info("Solicitud evaluada: %s", mensaje)


def main():
    """Configura logging e inicia la aplicación de escritorio."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(asctime)s - %(message)s",
    )
    FormularioCreditos().mainloop()


if __name__ == "__main__":
    main()
