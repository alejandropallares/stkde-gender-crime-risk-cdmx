"""
Script para geocodificar registros con coordenadas faltantes ("SIN REGISTRO")
usando la API de Geocoding de Google Maps.

Exporta el resultado en CSV con columnas de control:
  - ENVIADO_API: SI / NO
  - estado: vacío, para llenar manualmente con "correcto" o "incorrecto"

Uso:
    python geocodificar_dataset.py --input ruta/al/archivo.csv

Requiere un archivo .env con la variable GOOGLE_MAPS_API_KEY.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import googlemaps
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
import os

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

# Columnas del dataset (ajusta si tu archivo usa nombres ligeramente distintos)
COL_CALLE_1 = "CALLE 1 HECHOS"
COL_CALLE_2 = "CALLE 2 HECHOS"
COL_COLONIA = "COLONIA HECHOS"
COL_ALCALDIA = "ALCALDÍA HECHOS"
COL_COORD_X = "COORD. X"  # Longitud (WGS84)
COL_COORD_Y = "COORD. Y"  # Latitud (WGS84)
COL_ENVIADO_API = "ENVIADO_API"
COL_ESTADO = "estado"

# Marcador de coordenadas faltantes en el dataset original
SIN_REGISTRO = "SIN REGISTRO"
NO_ENCONTRADO = "NO ENCONTRADO"

# Guardado progresivo cada N registros geocodificados
INTERVALO_BACKUP = 50

# Pausa entre peticiones a Google (segundos) para respetar rate limits
PAUSA_ENTRE_PETICIONES = 0.1

# UTF-8 con BOM: Excel y la mayoría de herramientas en español lo reconocen correctamente
CODIFICACION_CSV_SALIDA = "utf-8-sig"
CODIFICACIONES_CSV_ENTRADA = ("utf-8-sig", "utf-8", "cp1252", "latin-1")

# Valores que NO representan calles, colonias u otros componentes válidos de dirección
VALORES_INVALIDOS = {
    "SIN REGISTRO",
    "SIN CALLES DE SAP",
    "SIN CALLES DEL SAP",
    "NO REFIERE",
    "NO ESPECIFICA LUGAR DE LOS HECHOS",
    "NO ESPECIFICA LUGAR DE HECHOS",
    "SIN LUGAR DE HECHOS",
    "SE DESCONOCE EL LUGAR DE LOS HECHOS",
    "NO PROPORCIONA CALLE DE HECHOS",
    "NO RECUERDA EL LUGAR (NO SCINCE)",
    "DESCONOCIDO",
    "NO SCINCE",
    "- NO SCINCE -",
    "NAN",
    "NONE",
    "",
}

# Patrones adicionales para descartar textos descriptivos que no son direcciones
PATRONES_INVALIDOS = [
    re.compile(r"^NO\s+ESPECIFICA", re.IGNORECASE),
    re.compile(r"^SIN\s+(CALLE|LUGAR|REGISTRO)", re.IGNORECASE),
    re.compile(r"^SE\s+DESCONOCE", re.IGNORECASE),
    re.compile(r"^NO\s+PROPORCIONA", re.IGNORECASE),
    re.compile(r"^NO\s+RECUERDA", re.IGNORECASE),
    re.compile(r"^EN\s+EL\s+DOMICILIO", re.IGNORECASE),
    re.compile(r"^NOTIF\.", re.IGNORECASE),
    re.compile(r"\(NO\s+SCINCE\)", re.IGNORECASE),
    re.compile(r"NO\s+ES\s+EL\s+LUGAR\s+DE\s+HECHOS", re.IGNORECASE),
]


def normalizar_texto(valor) -> str | None:
    """
    Limpia un valor de celda y lo devuelve como string válido,
    o None si está vacío o es un marcador inválido.
    """
    if pd.isna(valor):
        return None

    texto = str(valor).strip()
    if not texto:
        return None

    texto_upper = texto.upper()
    if texto_upper in VALORES_INVALIDOS:
        return None

    for patron in PATRONES_INVALIDOS:
        if patron.search(texto):
            return None

    return texto


def construir_direccion(
    calle_1,
    calle_2,
    colonia,
    alcaldia,
    ciudad: str = "Ciudad de México",
    pais: str = "México",
) -> str | None:
    """
    Construye una cadena de búsqueda amigable para Google Maps en México.

    Formato objetivo:
        "{calle_1} y {calle_2}, {colonia}, {alcaldia}, Ciudad de México, México"

    Omite componentes nulos o inválidos. Si no hay ningún componente útil,
    devuelve None.
    """
    c1 = normalizar_texto(calle_1)
    c2 = normalizar_texto(calle_2)
    col = normalizar_texto(colonia)
    alc = normalizar_texto(alcaldia)

    partes_calle: list[str] = []
    if c1 and c2:
        partes_calle.append(f"{c1} y {c2}")
    elif c1:
        partes_calle.append(c1)
    elif c2:
        partes_calle.append(c2)

    componentes: list[str] = []
    if partes_calle:
        componentes.append(", ".join(partes_calle))
    if col:
        componentes.append(col)
    if alc:
        componentes.append(alc)

    # Si solo tenemos alcaldía o colonia, aún puede ser geocodificable
    if not componentes:
        return None

    componentes.extend([ciudad, pais])
    return ", ".join(componentes)


def tiene_calle_colonia_alcaldia(calle_1, calle_2, colonia, alcaldia) -> bool:
    """
    True si el registro tiene los tres componentes mínimos para geocodificar:
    al menos una calle válida (calle 1 o calle 2), colonia y alcaldía.
    """
    c1 = normalizar_texto(calle_1)
    c2 = normalizar_texto(calle_2)
    col = normalizar_texto(colonia)
    alc = normalizar_texto(alcaldia)
    return bool(c1 or c2) and col is not None and alc is not None


def es_sin_registro(valor) -> bool:
    """Indica si una celda de coordenada está marcada como SIN REGISTRO."""
    if pd.isna(valor):
        return True
    return str(valor).strip().upper() == SIN_REGISTRO


def cargar_csv(ruta: Path, fila_encabezado: int = 0) -> pd.DataFrame:
    """
    Carga un CSV probando codificaciones comunes en español.
    Prioriza UTF-8 con BOM (utf-8-sig) para acentos y ñ.
    """
    ultimo_error: UnicodeDecodeError | None = None
    for encoding in CODIFICACIONES_CSV_ENTRADA:
        try:
            df = pd.read_csv(ruta, header=fila_encabezado, encoding=encoding)
            if encoding != CODIFICACION_CSV_SALIDA:
                print(f"  CSV leído con encoding: {encoding}")
            return df
        except UnicodeDecodeError as exc:
            ultimo_error = exc

    raise ValueError(
        f"No se pudo leer {ruta} con las codificaciones: {CODIFICACIONES_CSV_ENTRADA}"
    ) from ultimo_error


def cargar_dataset(ruta: Path, hoja: str | int | None = 0, fila_encabezado: int = 1) -> pd.DataFrame:
    """
    Carga CSV o Excel. Para el archivo de muestra de la CDMX,
    el encabezado real está en la fila 2 (índice 1).
    """
    extension = ruta.suffix.lower()
    if extension in {".xlsx", ".xls"}:
        return pd.read_excel(ruta, sheet_name=hoja, header=fila_encabezado)
    if extension == ".csv":
        return cargar_csv(ruta, fila_encabezado=fila_encabezado)
    raise ValueError(f"Formato no soportado: {extension}. Usa .csv, .xlsx o .xls")


def preparar_columnas_coordenadas(df: pd.DataFrame) -> pd.DataFrame:
    """Permite mezclar strings ('SIN REGISTRO') con coordenadas numéricas."""
    for col in (COL_COORD_X, COL_COORD_Y):
        df[col] = df[col].astype(object)
    return df


def preparar_columnas_control(df: pd.DataFrame) -> pd.DataFrame:
    """Inicializa columnas de trazabilidad para la geocodificación."""
    df[COL_ENVIADO_API] = "NO"
    df[COL_ESTADO] = ""
    return df


def geocodificar_direccion(cliente: googlemaps.Client, direccion: str) -> tuple[float | None, float | None, str]:
    """
    Consulta Google Maps Geocoding API.

    Retorna:
        (latitud, longitud, estado)
        estado puede ser: 'OK', 'SIN_DIRECCION', 'NO_ENCONTRADO', 'ERROR'
    """
    if not direccion:
        return None, None, "SIN_DIRECCION"

    try:
        resultados = cliente.geocode(
            direccion,
            components={"country": "MX"},
            region="mx",
        )

        if not resultados:
            return None, None, "NO_ENCONTRADO"

        ubicacion = resultados[0]["geometry"]["location"]
        return ubicacion["lat"], ubicacion["lng"], "OK"

    except googlemaps.exceptions.ApiError as exc:
        print(f"  [API ERROR] {exc}")
        return None, None, "ERROR"
    except googlemaps.exceptions.TransportError as exc:
        print(f"  [RED/TRANSPORTE] {exc}")
        return None, None, "ERROR"
    except Exception as exc:  # noqa: BLE001 — captura genérica solicitada para no detener el script
        print(f"  [ERROR INESPERADO] {exc}")
        return None, None, "ERROR"


def identificar_filas_sin_registro(df: pd.DataFrame) -> pd.Index:
    """Filas donde X o Y (o ambas) están marcadas como SIN REGISTRO."""
    mask_x = df[COL_COORD_X].apply(es_sin_registro)
    mask_y = df[COL_COORD_Y].apply(es_sin_registro)
    return df.index[mask_x | mask_y]


def identificar_filas_a_geocodificar(df: pd.DataFrame) -> pd.Index:
    """
    Filas elegibles para consultar la API: SIN REGISTRO en coordenadas
    y con calle + colonia + alcaldía válidas.
    """
    mask_sin_registro = df[COL_COORD_X].apply(es_sin_registro) | df[COL_COORD_Y].apply(es_sin_registro)
    mask_direccion_completa = df.apply(
        lambda fila: tiene_calle_colonia_alcaldia(
            fila.get(COL_CALLE_1),
            fila.get(COL_CALLE_2),
            fila.get(COL_COLONIA),
            fila.get(COL_ALCALDIA),
        ),
        axis=1,
    )
    return df.index[mask_sin_registro & mask_direccion_completa]


def guardar_csv(df: pd.DataFrame, ruta: Path) -> None:
    """Exporta CSV en UTF-8 con BOM para visualización correcta de español en Excel."""
    df.to_csv(
        ruta,
        index=False,
        encoding=CODIFICACION_CSV_SALIDA,
        lineterminator="\n",
    )


def guardar_backup(df: pd.DataFrame, ruta_backup: Path) -> None:
    """Guarda progreso intermedio en CSV."""
    guardar_csv(df, ruta_backup)
    print(f"\n  >> Backup guardado: {ruta_backup}")


def exportar_csv_final(df: pd.DataFrame, ruta: Path) -> None:
    """Exporta el dataset final en formato CSV."""
    guardar_csv(df, ruta)


def procesar_dataset(
    df: pd.DataFrame,
    cliente: googlemaps.Client,
    ruta_backup: Path,
    intervalo_backup: int = INTERVALO_BACKUP,
    limite: int | None = None,
) -> tuple[pd.DataFrame, set[int]]:
    """
    Geocodifica filas con SIN REGISTRO que tengan calle + colonia + alcaldía.
    El resto de registros con SIN REGISTRO se dejan sin modificar.

    Retorna el DataFrame actualizado y el conjunto de índices enviados a la API.
    """
    indices_sin_registro = identificar_filas_sin_registro(df)
    indices = identificar_filas_a_geocodificar(df)
    omitidos = len(indices_sin_registro) - len(indices)
    indices_enviados_api: set[int] = set()

    print(f"\nRegistros con SIN REGISTRO: {len(indices_sin_registro)}")
    print(f"Con calle + colonia + alcaldía (consulta a API): {len(indices)}")
    print(f"Omitidos (sin dirección completa, conservan SIN REGISTRO): {omitidos}")

    if limite is not None:
        indices = indices[:limite]
        print(f"MODO PRUEBA: procesando solo {len(indices)} registros.")

    total = len(indices)
    if total == 0:
        print("No hay filas elegibles para geocodificar. Nada que procesar.")
        return df, indices_enviados_api

    procesados_desde_ultimo_backup = 0

    for idx in tqdm(indices, desc="Geocodificando", unit="registro"):
        fila = df.loc[idx]
        indices_enviados_api.add(idx)
        df.at[idx, COL_ENVIADO_API] = "SI"

        direccion = construir_direccion(
            fila.get(COL_CALLE_1),
            fila.get(COL_CALLE_2),
            fila.get(COL_COLONIA),
            fila.get(COL_ALCALDIA),
        )

        print(f"\nFila {idx} | ID={fila.get('ID', 'N/A')}")
        print(f"  Dirección: {direccion or '(sin componentes válidos)'}")

        lat, lon, estado_api = geocodificar_direccion(cliente, direccion or "")

        if estado_api == "OK" and lat is not None and lon is not None:
            # El dataset original usa X=longitud, Y=latitud
            df.at[idx, COL_COORD_X] = lon
            df.at[idx, COL_COORD_Y] = lat
            print(f"  Resultado: OK -> lat={lat:.6f}, lon={lon:.6f}")
        elif estado_api == "SIN_DIRECCION":
            df.at[idx, COL_COORD_X] = NO_ENCONTRADO
            df.at[idx, COL_COORD_Y] = NO_ENCONTRADO
            print("  Resultado: NO ENCONTRADO (dirección vacía o inválida)")
        else:
            df.at[idx, COL_COORD_X] = NO_ENCONTRADO
            df.at[idx, COL_COORD_Y] = NO_ENCONTRADO
            print(f"  Resultado: NO ENCONTRADO ({estado_api})")

        procesados_desde_ultimo_backup += 1

        # Guardado progresivo
        if procesados_desde_ultimo_backup >= intervalo_backup:
            guardar_backup(df, ruta_backup)
            procesados_desde_ultimo_backup = 0

        time.sleep(PAUSA_ENTRE_PETICIONES)

    return df, indices_enviados_api


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Geocodifica registros con coordenadas 'SIN REGISTRO' vía Google Maps API. "
            "Solo consulta la API si el registro tiene calle + colonia + alcaldía válidas."
        ),
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Ruta al archivo CSV o Excel de entrada.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Ruta del CSV de salida (por defecto: <nombre>_geocodificado.csv).",
    )
    parser.add_argument(
        "--backup",
        default="dataset_backup_temporal.csv",
        help="Archivo CSV para guardado progresivo (default: dataset_backup_temporal.csv).",
    )
    parser.add_argument(
        "--header-row",
        type=int,
        default=1,
        help="Fila de encabezado en Excel/CSV (0-indexada). Default: 1 para el dataset CDMX.",
    )
    parser.add_argument(
        "--intervalo-backup",
        type=int,
        default=INTERVALO_BACKUP,
        help=f"Guardar backup cada N registros (default: {INTERVALO_BACKUP}).",
    )
    parser.add_argument(
        "--limite",
        type=int,
        default=None,
        help="Procesar solo los primeros N registros SIN REGISTRO (útil para pruebas).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Cargar API Key desde .env
    load_dotenv()
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not api_key:
        print(
            "ERROR: No se encontró GOOGLE_MAPS_API_KEY.\n"
            "Crea un archivo .env en la misma carpeta del script con:\n"
            "  GOOGLE_MAPS_API_KEY=tu_clave_aqui",
            file=sys.stderr,
        )
        return 1

    ruta_entrada = Path(args.input)
    if not ruta_entrada.exists():
        print(f"ERROR: No existe el archivo de entrada: {ruta_entrada}", file=sys.stderr)
        return 1

    ruta_salida = Path(args.output) if args.output else ruta_entrada.with_name(
        f"{ruta_entrada.stem}_geocodificado.csv"
    )
    if ruta_salida.suffix.lower() != ".csv":
        ruta_salida = ruta_salida.with_suffix(".csv")
    ruta_backup = Path(args.backup)

    print(f"Cargando dataset: {ruta_entrada}")
    df = cargar_dataset(ruta_entrada, fila_encabezado=args.header_row)
    print(f"Filas cargadas: {len(df)}")
    df = preparar_columnas_coordenadas(df)
    df = preparar_columnas_control(df)

    # Verificar columnas requeridas
    columnas_requeridas = [COL_CALLE_1, COL_CALLE_2, COL_COLONIA, COL_ALCALDIA, COL_COORD_X, COL_COORD_Y]
    faltantes = [c for c in columnas_requeridas if c not in df.columns]
    if faltantes:
        print(f"ERROR: Faltan columnas en el dataset: {faltantes}", file=sys.stderr)
        print(f"Columnas disponibles: {list(df.columns)}", file=sys.stderr)
        return 1

    cliente = googlemaps.Client(key=api_key)
    df, indices_enviados_api = procesar_dataset(
        df,
        cliente,
        ruta_backup,
        intervalo_backup=args.intervalo_backup,
        limite=args.limite,
    )

    # Guardado final en CSV
    exportar_csv_final(df, ruta_salida)
    print(f"\nProceso finalizado. Archivo exportado: {ruta_salida}")
    print(f"  -> Codificación: {CODIFICACION_CSV_SALIDA} (UTF-8 con BOM, compatible con Excel en español)")
    print(f"  -> {len(indices_enviados_api)} registro(s) marcados con ENVIADO_API=SI")
    print(f"  -> Columna '{COL_ESTADO}' lista para verificación manual (correcto / incorrecto)")

    # Resumen
    ok_count = df[COL_COORD_X].apply(lambda v: isinstance(v, (int, float)) or (
        isinstance(v, str) and v not in (SIN_REGISTRO, NO_ENCONTRADO) and _es_numero(v)
    )).sum()
    no_encontrado = (df[COL_COORD_X].astype(str).str.upper() == NO_ENCONTRADO).sum()
    sin_registro = (df[COL_COORD_X].astype(str).str.upper() == SIN_REGISTRO).sum()
    print(f"Resumen -> Con coordenadas numéricas: {ok_count} | NO ENCONTRADO: {no_encontrado} | SIN REGISTRO restante: {sin_registro}")

    return 0


def _es_numero(valor: str) -> bool:
    try:
        float(valor)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
