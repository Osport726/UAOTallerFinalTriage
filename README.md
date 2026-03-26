# Triage de Soporte TI mediante Clasificación de Imágenes

> Implementación con Visión Computacional, FastAPI y ejecución de modelo en CPU
> Proyecto final - Desarrollo de Proyectos de Inteligencia Artificial
> Universidad Autónoma de Occidente

---

## Integrantes

| Nombre | Código |
|---|---|
| Alexander Calambas Ramirez | 22602907 |
| Angelo Parra Cortez | 22506988 |
| Oscar Portela Ospina | 22507314 |
| Sebastian Torres Cabrera | 22507322 |

---

## Descripción general

En los help desks y áreas de soporte TI, el proceso de **triage inicial** es una de las actividades más repetitivas y propensas a errores. Los agentes deben leer manualmente descripciones de usuarios, interpretar capturas de pantalla y determinar el tipo de incidente para enrutarlo al equipo correcto. Este flujo es lento y susceptible a errores de enrutamiento que incrementan el **MTTR** (Mean Time To Resolution) y la carga operativa.

Este proyecto propone una solución basada en **Visión por Computadora con Transformers** para automatizar ese triage: el sistema recibe una captura de pantalla de un incidente TI, la clasifica en una categoría de soporte predefinida y sugiere el equipo responsable de atenderlo - sin intervención manual.

---

## Objetivo general

Desarrollar y desplegar una aplicación web y API HTTP, ejecutable en CPU, que clasifique imágenes de incidencias TI en categorías de soporte (triage visual) y entregue predicciones Top-K con probabilidades, alcanzando un **Macro-F1 >= 0.70** sobre el conjunto de evaluación, con manejo de errores, pruebas automatizadas y builds reproducibles mediante Docker.

---

## Objetivos específicos

- Definir el catálogo de categorías de soporte (mínimo 6) con reglas de enrutamiento por equipo responsable.
- Construir un dataset de demostración (20–50 imágenes por categoría) con capturas simuladas o públicas, garantizando ausencia de PII.
- Implementar el módulo de inferencia sobre **DeiT-Tiny** con un clasificador final ajustado a las categorías de soporte.
- Exponer la solución mediante **FastAPI**: endpoint `POST /api/predict` (Top-K con probabilidad) y `GET /health`.
- Construir una interfaz web (HTML/Jinja2) para carga de imágenes y visualización del resultado de enrutamiento.
- Incorporar pruebas automatizadas (mínimo 3) y pipeline CI en GitHub Actions para validación en cada push/PR.
- Empaquetar con **Docker** para despliegue reproducible y prueba de despliegue en DigitalOcean Droplet.

---

## Categorías de clasificación

El sistema clasifica imágenes en 8 categorías con enrutamiento sugerido al equipo responsable:

| Categoría | Ejemplos de imagen | Equipo sugerido |
|---|---|---|
| Red / Conectividad | Router, switch, Wi-Fi, errores de conexión | Equipo Redes |
| Acceso / Contraseñas | Pantallas de login, bloqueos, MFA | Soporte N1 / IAM |
| Correo / Office 365 | Outlook, OWA, errores de sincronización | Equipo O365 |
| Impresión / Periféricos | Impresoras, spooler, escáneres | Equipo Periféricos |
| Aplicación / Errores | Stack traces, errores de app, crash windows | Equipo Aplicaciones |
| Hardware / Equipo | PC sin encender, pantallas, componentes | Equipo Hardware |
| VPN / Remoto | Cliente VPN, errores de túnel, RDP | Equipo Seguridad/Redes |
| Otros / No clasifica | Casos raros o baja confianza | Revisión humana |

> **Regla de seguridad:** Si el score de confianza del modelo es menor a **0.40**, el sistema retorna automáticamente la categoría "Otros / Revisión humana", evitando enrutamientos automáticos con baja certeza.

---

## Stack tecnológico

| Componente | Tecnología |
|---|---|
| Backend API | FastAPI (async) + Uvicorn (ASGI) |
| Frontend | HTML/CSS + Jinja2 templates |
| Modelo de IA | DeiT-Tiny - HuggingFace Transformers + PyTorch CPU |
| Clasificador final | scikit-learn (StandardScaler + RandomForest 300 árboles), exportado como `.pkl` |
| Contenerización | Docker - imagen reproducible para CPU (`python:3.12-slim`) |
| CI/CD | GitHub Actions (lint + tests) + GitLab Runner (deploy a DigitalOcean) |
| Despliegue demo | DigitalOcean Droplet (Ubuntu 22.04 LTS) con Docker |

### Modelo de IA - DeiT-Tiny

Se utiliza **[DeiT-Tiny](https://huggingface.co/facebook/deit-tiny-patch16-224)** (Data-efficient Image Transformers, Facebook AI Research) como extractor de embeddings visuales. Arquitectura ViT con parches 16×16 px, imagen de entrada 224×224, preentrenado en ImageNet-1k (~22 MB en disco, ~200 MB en RAM).

Sobre los embeddings se entrena un pipeline `StandardScaler + RandomForest (300 árboles)` con `scikit-learn`, exportado como `.pkl`. La inferencia es 100% CPU, sin requerir GPU.

**Estrategia de respaldo:** Si no se alcanza F1 >= 0.70, se aplica data augmentation o se ajusta el clasificador. El valor real se documenta con justificación.

---

## Endpoints de la API

| Método | Endpoint | Descripción |
|---|---|---|
| `POST` | `/api/predict` | Recibe imagen (`multipart/form-data`), retorna Top-K categorías con score y equipo sugerido |
| `POST` | `/api/report` | Recibe imagen, genera y retorna un reporte PDF descargable con el resultado del triage |
| `GET` | `/health` | Healthcheck - retorna `{"status": "ok"}` con HTTP 200 en < 1s |

### Ejemplo de respuesta - `POST /api/predict`

```json
{
  "predictions": [
    {
      "category": "Red / Conectividad",
      "category_key": "red_conectividad",
      "score": 0.87,
      "team": "Equipo Redes"
    },
    {
      "category": "VPN / Remoto",
      "category_key": "vpn_remoto",
      "score": 0.08,
      "team": "Equipo Seguridad/Redes"
    },
    {
      "category": "Acceso / Contraseñas",
      "category_key": "acceso_contrasenas",
      "score": 0.03,
      "team": "Soporte N1 / IAM"
    }
  ],
  "model": {
    "model_id": "facebook/deit-tiny-patch16-224",
    "top_k": 3,
    "min_confidence": 0.40
  },
  "timestamp": "2026-03-09T14:32:10.123456"
}
```

> Si el score del top-1 es menor a `min_confidence` (0.40), el sistema retorna automáticamente la categoría **"Otros / Revisión humana"** para evitar enrutamientos automáticos con baja certeza.

---

## Requisitos funcionales

| ID | Requisito | Criterio de aceptación |
|---|---|---|
| RF-01 | Configurar FastAPI con HTML/Jinja2 | `uvicorn` arranca y renderiza rutas HTML vía Jinja2 |
| RF-02 | Endpoint `GET /health` | Responde `{"status": "ok"}` HTTP 200 en < 1s |
| RF-03 | Endpoint `POST /api/predict` | Acepta imágenes, rechaza formatos inválidos con 4xx, retorna `{category, score}` |
| RF-04 | Cargar modelo DeiT-Tiny real | Carga desde checkpoint, Macro-F1 >= 0.70 en dataset de evaluación |
| RF-05 | Interfaz web completa | Muestra categoría predicha, score y timestamp; errores sin traceback al usuario |
| RF-06 | Exportar reporte como PDF | Botón de descarga; PDF incluye imagen, categoría, score, timestamp y reglas de enrutamiento |
| RF-07 | Dockerfile para despliegue completo | `docker build` + `docker run` levanta UI + API; contenedor incluye el modelo |
| RF-08 | Pipeline CI en GitHub Actions | `push`/`PR` → lint + tests; build fallido bloquea merge |

## Requisitos no funcionales

| ID | Categoría | Descripción |
|---|---|---|
| RNF-01 | CPU / Infraestructura | Inferencia sin GPU; compatible con DigitalOcean Droplet de bajo costo |
| RNF-02 | Privacidad | Dataset de demo sin PII; capturas simuladas o públicas |
| RNF-03 | Confiabilidad | Respuestas 4xx controladas ante entradas inválidas; sin excepciones no manejadas |
| RNF-04 | Calidad | Mínimo 3 pruebas automatizadas y CI activo en cada push/PR |
| RNF-05 | Portabilidad | Dockerfile para ejecución en DigitalOcean Droplet / Linux VM sin modificaciones |
| RNF-06 | Desempeño | Macro-F1 >= 0.70 en el conjunto de evaluación; Top-1 routing accuracy reportada |

---

## Métricas de evaluación

- Accuracy
- Macro-F1 (umbral mínimo: **0.70**)
- Matriz de confusión
- Top-1 y Top-2 routing accuracy

---

## Fuera de alcance

- Resolución automatizada de incidentes (solo triage y enrutamiento).
- Integración productiva con plataformas ITSM (ServiceNow, Jira) - solo demo.
- Autenticación empresarial o manejo de usuarios.
- Datos reales con información personal identificable (PII).
- Entrenamiento a gran escala o modelos de producción de alto rendimiento.

---

## Dependencias del proyecto

### Producción

| Librería | Versión | Función |
|---|---|---|
| `torch` (CPU) | >= 2.0 | Motor de inferencia del modelo DeiT-Tiny |
| `transformers` | >= 4.35 | Carga de modelo y procesador de imágenes (HuggingFace) |
| `Pillow` | >= 10.0 | Preprocesamiento y validación de imágenes |
| `scikit-learn` | >= 1.3 | Clasificador final, métricas, matriz de confusión |
| `numpy` | >= 1.24 | Manipulación de arrays y embeddings |
| `fastapi` | >= 0.104 | Framework web para API REST y servicio HTML |
| `uvicorn` | >= 0.24 | Servidor ASGI para correr FastAPI |
| `python-multipart` | >= 0.0.6 | Soporte para carga de archivos en FastAPI |
| `jinja2` | >= 3.1 | Templates HTML para el frontend |
| `joblib` | >= 1.3 | Serialización y carga del clasificador `.pkl` |
| `reportlab` | >= 4.4 | Generación de reportes PDF del resultado de triage |

### Desarrollo

| Librería | Versión | Función |
|---|---|---|
| `pytest` | >= 7.0 | Ejecución de pruebas unitarias y de API |
| `httpx` | >= 0.25 | Cliente HTTP para pruebas de endpoints en pytest |
| `ruff` | >= 0.15 | Linter y formateador de código |
| `pre-commit` | >= 4.5 | Hooks de calidad antes de cada commit |

---

## Configuración del entorno

Este proyecto utiliza **uv** como gestor de dependencias y entornos virtuales. Requiere Python >= 3.10.

### Mac (Apple Silicon / Intel)

`make` viene preinstalado en macOS. Los comandos funcionan directamente desde la terminal:

```bash
make install # instala todas las dependencias
make start # levanta el servidor en http://localhost:8000
make test # corre los tests
make lint # verifica calidad del código
```

### Windows

`make` no está disponible en PowerShell por defecto. Hay dos opciones:

**Opción A - Git Bash (recomendada)**

Si tienen [Git for Windows](https://git-scm.com/) instalado, abrir **Git Bash** y usar los mismos comandos `make`:

```bash
make install
make start
make test
```

**Opción B - PowerShell / CMD (sin make)**

Usar `uv` directamente como equivalente:

```powershell
uv sync # equivale a make install
uv run uvicorn src.api.main:app --reload # equivale a make start
uv run pytest tests/ # equivale a make test
uv run ruff check src/ tests/ scripts/ # equivale a make lint
```

### Con Docker

El `Dockerfile` usa `python:3.12-slim` con soporte **multi-arch** (amd64 + arm64). Docker Desktop selecciona la arquitectura correcta automáticamente — el mismo comando funciona en Mac (Intel y Apple Silicon) y Windows.

```bash
# Construir la imagen (incluye modelo entrenado y caché de DeiT-Tiny)
docker build -t triage-ti .

# Ejecutar el contenedor con el modelo real
docker run -p 8000:8000 triage-ti

# Ejecutar en modo desarrollo (sin modelo entrenado)
docker run -p 8000:8000 -e USE_DUMMY_MODEL=1 triage-ti
```

**Con Docker Compose** (levanta la app + servidor gRPC juntos):

```bash
# Producción — usa modelo real por defecto
docker compose up

# Desarrollo — sin modelo entrenado
USE_DUMMY_MODEL=1 docker compose up
```

La aplicación queda disponible en `http://localhost:8000`.

---

## Dataset

El proyecto construye un dataset de demostración propio con imágenes públicas y capturas simuladas, organizado en las 8 categorías de soporte. Ninguna imagen contiene PII.

- **Fuentes:** capturas simuladas, imágenes públicas (Unsplash, Pixabay, Wikimedia), screenshots genéricos de SO y aplicaciones en modo demo.
- **Volumen:** 20–50 imágenes por categoría (160–400 en total).
- **Estructura de carpetas:**

```
data/
├── raw/                    # Dataset completo, una carpeta por categoría
│   ├── red_conectividad/
│   ├── acceso_contrasenas/
│   ├── correo_office365/
│   ├── impresion_perifericos/
│   ├── aplicacion_errores/
│   ├── hardware_equipo/
│   ├── vpn_remoto/
│   └── otros/
└── test/                   # Generado automáticamente por scripts/train.py
    └── {categoria}/        # 15 % del dataset, separado para evaluación final
```

- **Split estratificado por categoría:**
  - 70 % entrenamiento
  - 15 % validación (usado durante el entrenamiento para selección del modelo)
  - 15 % test (separado en `data/test/`, usado exclusivamente por `scripts/evaluate.py`)

> El conjunto de test **nunca se usa durante el entrenamiento ni la validación**. `scripts/train.py` lo copia automáticamente a `data/test/` al final de cada ejecución.

---

## Pipeline de ML

### 1. Recolectar imágenes

Agregar capturas a `data/raw/{categoria}/` (`.jpg`, `.jpeg`, `.png` o `.webp`). Mínimo 20 imágenes por categoría para que el split 70/15/15 funcione con `stratify`.

### 2. Entrenar el modelo

```bash
make train
# o directamente:
uv run python scripts/train.py
```

El script:
1. Extrae embeddings de todas las imágenes usando DeiT-Tiny (192 dimensiones).
2. Divide el dataset en **70 % train / 15 % validación / 15 % test** (estratificado).
3. Copia las imágenes de test a `data/test/` para uso posterior de `evaluate.py`.
4. Entrena un pipeline `StandardScaler + RandomForest (300 árboles)` sobre los embeddings.
5. Guarda el modelo en `models/classifier.pkl` y las etiquetas en `models/labels.json`.
6. Registra métricas y artefactos en MLflow (`mlruns/`).

> Para probar otros clasificadores: `uv run python scripts/train.py --clf svm_rbf_c10` (opciones: `svm_rbf_c10`, `rf`, `rf500`, `et`, `gbt`, entre otros).

### 3. Evaluar en el conjunto de test

```bash
make evaluate
# o directamente:
uv run python scripts/evaluate.py
```

Lee las imágenes de `data/test/`, extrae embeddings y genera el reporte final (accuracy, Macro-F1, matriz de confusión). Este resultado es el que se reporta como métrica oficial del proyecto.

### 4. Ejecutar la aplicación con el modelo entrenado

```bash
make start
```

Una vez que `models/classifier.pkl` y `models/labels.json` existen, la app carga el modelo real automáticamente.

### Modo de desarrollo sin modelo entrenado

Para desarrollar la UI o los endpoints sin necesitar un modelo entrenado:

```bash
USE_DUMMY_MODEL=1 make start
# o en Windows PowerShell:
$env:USE_DUMMY_MODEL="1"; uv run uvicorn src.api.main:app --reload
```

En este modo, el sistema devuelve predicciones fijas de prueba sin cargar DeiT-Tiny.

---

## Cronograma

| ID | Tarea | Fechas | Entregable |
|---|---|---|---|
| T01 | Catálogo de categorías y reglas de enrutamiento | 26/02 | Documento validado |
| T02 | Recolección y etiquetado del dataset | 26/02 – 28/02 | Dataset en repo |
| T03 | Configuración entorno: repo + Dockerfile base | 26/02 – 27/02 | Repositorio en GitHub |
| T04 | Módulo de inferencia DeiT-Tiny + clasificador | 27/02 – 03/03 | Script de inferencia funcional |
| T05 | Endpoints FastAPI (`/api/predict` + `/health`) | 02/03 – 03/03 | Servicio probado con curl |
| T06 | Prueba de baseline y ajuste de umbrales | 03/03 – 04/03 | Reporte (accuracy, F1) |
| T07 | Interfaz web HTML/Jinja2 | 04/03 – 05/03 | UI funcional en navegador |
| T08 | Mejora del modelo si F1 < 0.70 | 04/03 – 06/03 | Modelo con Macro-F1 >= 0.70 |
| T09 | Pruebas automatizadas (unitaria, API, integración) | 05/03 – 06/03 | Tests pasando localmente |
| T10 | Pipeline CI GitHub Actions | 06/03 – 09/03 | Workflow funcional |
| T11 | Dockerfile final + prueba imagen Docker en CPU | 06/03 – 09/03 | Contenedor lanzado exitosamente |
| T12 | Evidencias, demo y organización del repositorio | 10/03 – 12/03 | Proyecto funcional y documentado |

**Ruta crítica:** T03 → T04 → T05 → T07 → T11 → T12

### Hitos

| Hito | Fecha | Criterio |
|---|---|---|
| Modelo entrenado | 03/03/2026 | DeiT-Tiny adaptado para las 8 categorías |
| Baseline funcional | 04/03/2026 | FastAPI accesible desde curl o Postman |
| App completa + CI | 09/03/2026 | UI funcional, tests pasando, Docker construido |
| Proyecto listo para entrega | 12/03/2026 | Aplicación desplegada en contenedor Docker |

---

## Metodología

El proyecto sigue una **metodología Ágil** con un sprint de desarrollo (26/02 – 09/03) más una fase de cierre (10/03 – 12/03). Cada tarea produce un incremento funcional entregable. El tablero Kanban del equipo está disponible en [GitHub Projects](https://github.com/users/sebasbit/projects/1/views/1).

---

## Licencia

MIT License - ver [LICENSE](LICENSE) para más detalles.
