# Auditoría técnica v2 — F1 Championship Predictor

**Fecha:** 2026-08-03
**Commit auditado:** `ef3b3c0` (main, tras PR #1 `audit-remediation` y PR #2 `feat/streamlit-cloud-entrypoint`)
**Alcance:** todo lo que `docs/AUDIT.md` (2026-07-30) no cubría o dio por cerrado — feature engineering,
inferencia/API/dashboard, tests/CI/Docker, modelado/calibración/explicabilidad — más verificación
independiente de que la remediación de `docs/ROADMAP.md` se aplicó como se describe en los commits.
**Método:** lectura directa del código actual, nunca de los mensajes de commit o la documentación por
sí solos; varios hallazgos se reprodujeron empíricamente (reinyectando código ya "corregido" contra la
suite de tests, o recalculando correlaciones sobre `data/processed/dataset_32abb3e1b8a6-2026-08-02.parquet`).

---

## Resumen ejecutivo

La ronda de remediación de `docs/ROADMAP.md` es real y la mayoría de sus puntos están genuinamente
cerrados: unificación de población (E3/E4), selección con incertidumbre (P1/P2), features colineales
exactas eliminadas (P3/E2), denominadores de tasas (E5), flag in-sample/out-of-sample (D2). Esto no es
una lectura complaciente — se verificó línea por línea y, en dos casos, reproduciendo el bug original
contra el código actual para confirmar que de verdad ya no se dispara.

Lo nuevo, ordenado por lo que más pesaría en una decisión de contratación o de despliegue:

1. **No hay autenticación en ningún endpoint que cambia estado** (`/train`, `/update-data`,
   `/model/{id}/activate`), y el proyecto acaba de añadir un entrypoint público de Streamlit Cloud.
2. **`momentum` reproduce, en el mismo commit que arregló `season_points_rate`, el mismo patrón de bug**
   (divide por el tamaño fijo de la ventana, no por rondas realmente disputadas) — afecta a las rondas
   2-3 de toda temporada, no solo a los debutantes a mitad de año que motivaron el fix original.
3. **La suite `e2e` está vacía** pese a documentarse como categoría de test activa.
4. El guardarraíl de eliminación matemática **falla en abierto** si el bloque de features que lo produce
   se desactiva.
5. Condición de carrera sin bloqueo en las cachés de `PredictionService` bajo peticiones concurrentes.
6. El flujo `nightly-update.yml` reentrena dentro de una caché efímera de GitHub Actions, sin conexión
   con ningún despliegue real.
7. Cero búsqueda de hiperparámetros en los 8 candidatos del "model zoo", no declarado en ningún sitio.

Nada de esto invalida el trabajo de remediación anterior. Es la siguiente capa del mismo patrón: el
proyecto verifica bien lo que va a buscar, y dos de los hallazgos de aquí (el bug de `momentum` y la
sobreestimación de cobertura del fixture de `CircuitBlock` en el propio `docs/ROADMAP.md`) muestran que
la verificación narrada todavía va un paso por delante de la verificación real.

**Actualización 2026-08-03 (misma sesión):** N1, N3, N4, N6, N7, N8 y N9 están corregidos y
regresión-testeados (ver la sección de cada uno). N5 ya estaba resuelto de forma independiente por el
autor (`feat/api-read-only-mode`, ya en `main`) con un enfoque más adecuado que el que se iba a proponer
aquí — ver su entrada. Quedan abiertos: N2 (test dedicado de `CircuitBlock`) y los puntos de R1-R6 y de
MLOps, todos de menor prioridad inmediata o de mayor coste/riesgo. Pendiente de reconstruir el dataset y
reentrenar para que el fix de `momentum` (N1) y el cambio de features (N8/N9) se reflejen en un modelo
real.

---

## Estado de la remediación de `docs/ROADMAP.md`

| Punto del roadmap | Estado verificado | Evidencia |
|---|---|---|
| 1. Unificar población puntuada (E3/E4) | **FIXED** | `dataset/builder.py:159-191` ya no filtra `entered=False`; `models/normalizer.py:120-140` zera `is_mathematically_eliminated` antes de renormalizar en el único punto de entrada compartido |
| 2. Podar SVM/k-NN | **FIXED** | `conf/models.yaml:159-177`, `enabled: false` en ambos |
| 3. Selección 1-SE | **FIXED, correctamente implementado** | `training/selection.py:114-181`, verificado que no cambia el ganador pero sí lo justifica (LightGBM entra en el conjunto empatado) |
| 4. Flag in-sample/out-of-sample | **PARCIAL** | Correcto en `predictor.py`, `predict.py`, y en 3 de 5 páginas del dashboard que reproducen histórico — ver N4 |
| 5. Sanear métricas (E6/E7/P4/P5) | **FIXED** | `precision`/`recall`/`f1` eliminadas de `evaluation/metrics.py`; baseline junto a cada métrica; ECE agrupado sobre pool en vez de promedio de 30 ECEs (`trainer.py:610-642`) |
| 6. Eliminación como restricción, no feature (D3) | **FIXED, pero frágil** | Ver N3 |
| 7. Denominadores de tasas (E5) | **FIXED y regresión-testeado** | `rates.py:59-61`, `form.py:74-82`; reintroducir el bug original hace fallar el test |
| 8. Higiene de features (P3/E2) | **FIXED para los pares exactos, incompleto para pares nuevos** | Ver N7 |
| 9. Ampliar fixture de leakage (D5) | **PARCIAL — la afirmación del roadmap es más fuerte de lo que el código demuestra** | Ver N2 |
| 10. `CircuitBlock`: arreglar, medir, decidir (E1) | **FIXED el join; nunca medido su aporte real** | Nulls: 99.68% → 42.6%. No hay ningún test ni medición del delta de `group_log_loss` que el propio roadmap pedía |
| 11. Tabla de ablación en README (D6) | **FIXED** | README incluye la tabla top-5 vs. 73 features |
| D1 (refit sin 2023-2025) | **Documentado, no resuelto** — decisión consciente | Ver N8 |
| D4 (features constantes por grupo) | **Documentado correctamente, no tocado** — decisión correcta | `models/normalizer.py:25-33` |
| D7 (`min_train_seasons` semántica) | **Documentado con un comentario, no resuelto** | `conf/models.yaml:14-17` |

---

## Hallazgos nuevos confirmados

### N1 — `momentum`/`_recent_rate` divide por el tamaño fijo de ventana, no por rondas disputadas [FIXED 2026-08-03]

Corregido en `features/form.py`: `_recent_rate` ahora divide por rondas realmente disputadas dentro de
la ventana (mismo patrón que `season_points_rate`), no por el tamaño fijo de la ventana. Test de
regresión: `tests/unit/features/test_pipeline.py::test_momentum_divides_by_rounds_started_not_window_size`
(asserta `momentum == 0.0` en la segunda ronda de un debutante). Requiere reconstruir el dataset y
reentrenar para que el modelo servido refleje el fix — ver la sección de reentrenamiento al final.

**Archivo:** `src/f1cp/features/form.py:84-91`

```python
(pl.col(f"last_{block_config.windows[0]}_points") / block_config.windows[0]).alias("_recent_rate"),
...
(pl.col("_recent_rate") - pl.col("season_points_rate")).alias("momentum"),
```

`last_3_points` es una suma móvil que respeta cuántas rondas existen realmente en la ventana (1, 2 o 3).
`_recent_rate` la divide por la constante literal `3` sin importar cuántas rondas contribuyeron.
`season_points_rate`, dos líneas más arriba, **en el mismo commit** (`51a4b53`, el que arregló E5), sí se
corrigió para dividir por rondas realmente disputadas. `_recent_rate` no.

**Evidencia (dataset real, `data/processed/dataset_32abb3e1b8a6-2026-08-02.parquet`):**
`momentum + season_points_rate == last_3_points/3` se cumple en las **19.764 filas no nulas** sin
excepción. Ejemplo: Alan Jones, 1981 ronda 2 (su segunda carrera, ganó la primera):
`season_points_rate = 9.0` (correcto), `momentum = -6.0` (falso: implica un declive de forma cuando
"reciente" y "de temporada" son la misma carrera). Reproducido de forma idéntica en el fixture sintético:

```
round  last_3_points  season_points_rate  momentum
2      15.0           15.0                -10.0   # debería ser 0.0
3      33.0           16.5                -5.5    # debería ser 0.0
4      58.0           19.33               0.0     # ventana llena — correcto
```

**Alcance:** no son los debutantes a mitad de temporada de E5 (16% de temporadas-piloto) — es **todo
piloto, toda temporada**, en las rondas 2 y 3, justo la fase temprana que el proyecto ya reconoce como
la más difícil.

**Cobertura de test:** `tests/unit/features/test_form.py` solo prueba `_rolling_ols_slope` (`trend`).
No existe ningún test de `momentum`, `_recent_rate` ni `consistency`.

**Gravedad: Alta.**
**Cómo arreglarlo:** sustituir `block_config.windows[0]` por el mismo denominador de rondas-realmente-
disputadas ya usado en `season_points_rate`/`rates.py` (recuento acumulado de `entered`), y añadir un
test de regresión análogo al de `season_points_rate`. Requiere reconstruir el dataset y reentrenar
(afecta a una feature real, no solo a metadatos).

---

### N2 — La cobertura de test que reclama `docs/ROADMAP.md` (punto 9) es solo parcialmente cierta [FIXED 2026-11-22]

`tests/unit/features/test_circuit.py` compara `CircuitBlock` con una referencia fila a fila y fija un valor
calculado a mano. Con el join roto reintroducido, sus tres tests fallan (el de leakage sigue pasando,
como se midió abajo).

`docs/ROADMAP.md` afirma que ampliar `synthetic.py` (debutante a mitad de temporada, ausencia, circuito
repetido) "habría cazado E1 y E5". Verificado empíricamente reintroduciendo el join roto original de
`CircuitBlock` (`on=["season","round","driver_id","circuit_id"]`) contra el fixture actual:

```
OLD buggy circuit feature null rate on synthetic fixture: 1.0   (100% null)
LEAKAGE TEST: PASSED (bug NOT detected)
```

`test_features_are_unchanged_by_deleting_the_future` pasa sin cambios con el join roto reinstalado, y no
existe ningún `test_circuit*.py` que compruebe valores o no-nulidad de `avg_finish_at_remaining_circuits`.
El mismo experimento para el bug de denominadores de `rates.py` **sí falla** como se espera — ese caso
está genuinamente cubierto.

**Gravedad: Media.**
**Cómo arreglarlo:** un test dedicado de `CircuitBlock` que fije valores concretos sobre el escenario de
circuito repetido que el propio fixture ya contiene.

---

### N3 — El guardarraíl de eliminación matemática falla en abierto [FIXED 2026-08-03]

`normalize_round_probabilities` ahora lanza `ValueError` si `is_mathematically_eliminated` no está en
el frame y la normalización está activada, en vez de aplicar la restricción en silencio solo si la
columna existe. Tests: `tests/unit/models/test_normalizer.py::test_normalize_round_probabilities_requires_the_elimination_column_when_enabled`
y `..._zeroes_eliminated_drivers_before_rescaling`.

**Archivo:** `src/f1cp/models/normalizer.py:125`

```python
if ELIMINATED_COLUMN in frame.columns:
    constrained = frame.with_columns(...)
```

La restricción que evita el caso Bottas (76% de probabilidad sobre un piloto matemáticamente eliminado)
solo se aplica si la columna existe. La produce el bloque `season_context`, que es **independientemente
desactivable** en `conf/features.yaml`. Si se desactiva, o un futuro refactor renombra la columna, el
guardarraíl no falla ni avisa — simplemente deja de aplicar la restricción, y el pipeline vuelve
exactamente al comportamiento pre-fix, sin ningún test que lo detecte.

**Gravedad: Media.**
**Cómo arreglarlo:** `assert ELIMINATED_COLUMN in frame.columns` cuando `training.normalisation.enabled`
sea `true`, con un test que lo cubra.

---

### N4 — `home.py` y `feature_importance.py` no muestran el aviso in-sample/out-of-sample [FIXED 2026-08-03 para home.py]

`home.py` ya muestra el mismo aviso que `predictor.py`. Test:
`tests/integration/test_dashboard.py::test_home_page_warns_when_the_overview_is_in_sample`.
`feature_importance.py` se dejó sin tocar deliberadamente: muestra atribuciones SHAP, no probabilidades
de título, así que el riesgo de sobreestimación de rendimiento que motiva este aviso no aplica igual —
puede revisarse aparte si se decide que también lo necesita.

**Archivos:** `src/f1cp/dashboard/pages/home.py`, `src/f1cp/dashboard/pages/feature_importance.py`

El aviso está correctamente cableado en `predictor.py:64-70`, `driver_analysis.py:96-103` y
`historical_predictions.py:88-94`, pero `home.py` (la página de aterrizaje, la de más tráfico) llama a
`cached_predict` (línea 96) y renderiza `latest["probability"]`/`latest["drivers"]` (líneas 132-155) sin
leer nunca `latest["is_out_of_sample"]`. Hoy es invisible porque la ronda por defecto (la más reciente de
la temporada 2026 en curso) da la casualidad de ser genuinamente out-of-sample — pero no está garantizado
por nada: en cuanto la ingesta de la temporada en vivo se retrase, o se despliegue una demo fresca
sembrada solo hasta 2025, la página resolverá a una temporada dentro de la ventana de calibración,
`is_out_of_sample` pasará a `False`, y la página más visible del producto mostrará una repetición de una
temporada que el modelo ya conoce, sin ningún aviso — mientras tres páginas hermanas a un clic avisan
correctamente sobre el mismo dato subyacente.

**Gravedad: Media-Alta** (es la página de más tráfico y la única que el fix se olvidó).
**Cómo arreglarlo:** replicar el mismo patrón de aviso de `predictor.py` en `home.py`, y considerar si
`feature_importance.py` necesita la misma nota (menor prioridad: muestra atribuciones, no probabilidades).

---

### N5 — Sin autenticación en ningún endpoint que cambia estado [FIXED de forma independiente, 2026-08-03]

Resuelto por el autor, fuera de esta sesión de correcciones, con un enfoque más adecuado que una API key:
un modo `api.read_only` (`API_READ_ONLY`) que devuelve 403 en `/train`, `/update-data` y
`/model/{id}/activate` sin afectar a las rutas de lectura. `streamlit_app.py` lo activa solo en el
despliegue público; Docker/local siguen sin restricción por ser de un solo usuario. El dashboard oculta
los controles en vez de mostrar un botón muerto (`/health` expone `read_only`). Test:
`tests/integration/test_api.py::test_read_only_mode_blocks_mutating_endpoints_but_not_reads`.
No es autenticación por credencial (no distingue admin de visitante), pero para una única instancia
pública compartida es la solución de menor complejidad que cierra el vector de abuso real.

**Archivos:** `src/f1cp/api/main.py`, `src/f1cp/api/routers/{train,data,model}.py`,
`src/f1cp/dashboard/pages/settings.py`

`POST /train`, `POST /update-data` y `POST /model/{id}/activate` son alcanzables sin ningún control de
acceso — no hay middleware de auth en `api/main.py` ni `api/deps.py`. No es teórico: el proyecto acaba
de añadir un entrypoint público de **Streamlit Community Cloud** (`streamlit_app.py`), y
`dashboard/pages/settings.py` pone directamente en la UI pública botones de "lanzar reentrenamiento
completo" y "bootstrap de datos". Cualquier visitante puede disparar repetidamente un reentrenamiento (30
folds × 8 candidatos, minutos de CPU) o un bootstrap completo (recorre décadas de la API externa), sin
límite de tasa ni registro de quién lo originó más allá de un UUID.

**Gravedad: Crítica** para cualquier despliegue accesible desde internet.
**Cómo arreglarlo:** como mínimo, una API key simple en un header para los tres endpoints de escritura,
o separar un modo "solo lectura" (predict/explain/standings) del panel de administración
(train/update-data/activate), que no debería exponerse fuera de una red de confianza.

---

### N6 — `/model/{id}/activate` no valida que la versión sea servible [FIXED 2026-08-03]

Validado en el router de la API (`api/routers/model.py::activate`), no en
`ModelVersionRepository.activate`: el repositorio lo sigue usando el trainer para activar su propio
ganador justo después de entrenar, y ese camino debe seguir funcionando aunque el tracking de MLflow esté
desactivado o degrade de forma no estricta (ver `training/tracking.py`). El endpoint HTTP, que es por
donde un operador activa una versión arbitraria, ahora responde 409 si `mlflow_model_uri` es `None`. Test:
`tests/integration/test_api.py::test_activating_a_version_with_no_logged_model_is_rejected`.

**Archivo:** `src/f1cp/db/repositories/mlops.py:148-172`

`ModelVersionRepository.activate` solo comprueba que la fila existe, nunca que `mlflow_model_uri is not
None`. Una versión con logging de MLflow fallido (o `MLFLOW_ENABLED=false`) puede existir en la tabla y
aparecer sin filtrar en el desplegable de `model_performance.py:158-168`. Activarla tiene éxito (200 OK);
cada `/predict` posterior devuelve un 404 (`predictor.py:425-427`). `/health`
(`api/routers/health.py:30-32`) solo comprueba que existe un modelo activo, no que sea servible, así que
el servicio queda caído sin ninguna señal de salud hasta que alguien llama a `/predict`.

**Gravedad: Alta.**
**Cómo arreglarlo:** validar `mlflow_model_uri is not None` (y, si es posible, que `feature_names`
coincide con la configuración actual) antes de activar; que `/health` verifique servibilidad, no solo
existencia.

---

### N7 — Condición de carrera sin bloqueo en las cachés de `PredictionService` [FIXED 2026-08-03]

Cada caché (`_panel_cache`, `_explain_cache`, `_pipeline_cache`) tiene ahora su propio `threading.Lock`
alrededor de la sección de lectura/evicción/inserción. La construcción cara (rebuild del panel, SHAP,
carga de MLflow) se hace deliberadamente fuera del lock, para no serializar peticiones concurrentes en la
más lenta — un fallo simultáneo en la misma clave reconstruye dos veces en vez de bloquear, que es un
coste aceptable frente al `KeyError` no capturado que había antes.

**Archivo:** `src/f1cp/inference/predictor.py` (`_panel_cache`, `_pipeline_cache`, `_explain_cache`)

Son diccionarios sin bloqueo, compartidos por una única instancia de `PredictionService` a través de toda
la app. Las rutas de FastAPI que los usan son `def` síncronas, no `async def`, así que Starlette las
despacha a hilos reales del threadpool — concurrencia genuina. La secuencia de invalidación
(`for stale in [...]: del self._panel_cache[stale]`) no es atómica: dos peticiones concurrentes que
observen el mismo cambio de `source_data_revision` pueden intentar borrar la misma clave dos veces, y la
segunda `del` lanza `KeyError` sin capturar → 500 sin relación con lo que pidió el usuario.

**Gravedad: Alta.**
**Escenario concreto:** un `/update-data` en segundo plano termina mientras hay tráfico simultáneo del
dashboard en dos pestañas — ambos hilos compiten por invalidar el mismo panel.
**Cómo arreglarlo:** un `threading.Lock` alrededor de la sección crítica de invalidación/inserción, o una
estructura de caché con operaciones atómicas.

---

### N8 — Nuevo par casi perfectamente colineal no detectado por la pasada de higiene de features [FIXED 2026-08-03, con un motivo más fuerte del que se pensaba]

`position_order` se movió a `metadata_columns` en `conf/features.yaml` (ya no es una feature del modelo).
No se eliminó solo por la colinealidad con `finish_position` (r=0.9998) — esa cifra, por sí sola, es del
mismo orden que pares que el proyecto decidió **no** podar (`career_starts`/`experience_seasons`
r≈0.97), y hacerlo habría sido selección manual de features sin más justificación que la correlación,
justo lo que `docs/ROADMAP.md` advierte no hacer. La razón real, encontrada al revisar los proveedores:
para una fila de abandono (DNF), `position_order` **no significa lo mismo según qué proveedor la
ingirió** — `f1db` reporta el orden de clasificación real (`positionDisplayOrder`), `jolpica` usa un
valor fijo `99` (`data/providers/jolpica.py:316,360`) cuando no hay posición. Una feature cuyo
significado para el mismo tipo de evento depende de qué proveedor rellenó esa ronda es ruido, no señal
que un modelo deba poder ponderar. Sigue disponible como columna de metadatos para las páginas de
archivo, que la leen directamente de la base de datos, no del panel de features.

Correlaciones calculadas sobre el dataset real (referencia, no todas de acción inmediata):

```
finish_position / position_order        r = 0.9998  (n=12.331)  -- resuelto, ver arriba
grid / quali_position                   r = 0.9869  (n=17.876)
season_avg_grid / last_3_avg_grid       r = 0.9623  (n=18.099)
last_3_avg_finish / last_5_avg_finish   r = 0.9597  (n=15.804)
```

Los tres pares restantes (r entre 0.96 y 0.99) se dejan sin tocar, igual que
`career_starts`/`experience_seasons` (r≈0.97) y `career_wins`/`prior_titles` (r≈0.958) ya documentados
como decisión correcta en `docs/ROADMAP.md`: son correlación, no una identidad algebraica, y la
regularización L2 ya existente los maneja mejor que una poda manual sin más evidencia que el coeficiente
de correlación.

---

### N9 — `teammate_beat_rate` compara puntos de ronda, no posición de llegada [FIXED 2026-08-03]

Ahora compara `finish_position` (con un valor centinela para abandonos, peor que cualquier posición
clasificada, así que un DNF nunca "gana" a un compañero clasificado y dos DNFs en la misma ronda son un
empate). Tests en `tests/unit/features/test_constructor.py` (nuevo archivo): ronda empatada a puntos
decidida por posición, DNF vs. clasificado, y doble DNF como empate.

**Archivo:** `src/f1cp/features/constructor.py:159-163`

```python
(pl.col("round_points") > pl.col("_teammate_round_points")).mean().alias("_beat_teammate_this_round"),
```

El propio docstring del bloque la llama "una de las señales más fuertes disponibles para contención de
título". Pero en cualquier ronda donde ambos compañeros de equipo terminen fuera de puntos (frecuente en
la zona media/baja de parrilla), la comparación es `False > False = False` para los dos, aunque hubiera
una diferencia real de posición (P11 vs. P16). La feature solo es sensible a quién superó a quién en la
minoría de rondas donde al menos uno puntúa.

**Gravedad: Media.**
**Cómo arreglarlo:** comparar `finish_position`/`position_order` (o el orden de llegada, que sí maneja
abandonos) en vez de puntos de ronda.

---

## Riesgos y decisiones discutibles

### R1 — Cero búsqueda de hiperparámetros, no declarado en ningún documento

`grep` exhaustivo sobre `src/`, `conf/`, `docs/`, `pyproject.toml` por
`GridSearchCV|RandomizedSearchCV|optuna|hyperopt|BayesSearchCV` no encuentra nada. Los 8 algoritmos de
`conf/models.yaml` tienen hiperparámetros fijos elegidos a mano (razonables, pero nunca validados por una
búsqueda). Ni el README ni `CLAUDE.md` aclaran esto en ningún punto.

**Por qué importa:** con el top-3 de modelos ya indistinguible estadísticamente (LR vs. LightGBM
p=0.31), es plausible que parte de esa indistinguibilidad venga de que ninguna familia ha tenido su
oportunidad real, no de que el problema sea igual de difícil para las tres.

**Gravedad: Media-Alta.**
**Cómo arreglarlo:** una búsqueda acotada (Optuna/`RandomizedSearchCV`) sobre 2-3 candidatos dentro de
cada fold externo, o como mínimo una frase explícita de limitación en el README.

### R2 — El modelo final desplegado nunca se reentrena sobre las últimas 3 temporadas completas [MEDIDO 2026-08-03]

Documentado y argumentado en `training/trainer.py:1-17` (recuperar 3 de 45 etiquetas no compensa la
complejidad de una calibración cross-fitted) — decisión razonable *como trade-off*, pero no estaba
cuantificada. Medido reduciendo `calibration_holdout_seasons` sobre los mismos 30 folds
(`logistic_regression`, 72 features, todo lo demás igual):

| `calibration_holdout_seasons` | Temporadas en el ajuste | group log-loss |
|---|---|---|
| **3 (actual)** | 1981–2022 (42) | **0.615** |
| 2 | 1981–2023 (43) | 0.654 |
| 1 | 1981–2024 (44) | 0.690 |
| 0 (sin calibrar) | 1981–2025 (45) | 0.706 |

Darle al ajuste temporadas más recientes **empeora** el resultado de forma monótona, porque el
calibrador se queda con menos etiquetas independientes (3→2→1→0) y ese ruido pesa más que las filas de
entrenamiento ganadas. La configuración actual (holdout=3) es la mejor de las cuatro probadas —
no es un compromiso que esté costando rendimiento sin necesidad.

**Lo que esto no descarta:** esto mide la versión barata (quitarle temporadas a la calibración para
dárselas al ajuste), no la calibración cross-fitted de verdad (K-fold interno sobre temporadas +
reajuste del modelo base sobre out-of-fold), que en teoría podría mejorar sobre 0.615 sin este
compromiso. No implementada por su coste de complejidad frente al beneficio incierto — decisión
tomada con el dato de esta tabla delante, no solo por argumento.

**Gravedad: Baja** (bajada de Media-Alta: ya no es un riesgo sin cuantificar, es una decisión medida).

### R3 — Tres correcciones que se solapan sin validar su necesidad conjunta

`class_weight="balanced"` + calibración Platt + normalización por grupo son, cada una, correctas por
separado (verificado sin fuga: el peso de clase se calcula solo sobre el slice de ajuste). Pero no existe
ninguna ablación que compare `imbalance.strategy: balanced` contra `none` **después** de calibración y
normalización sobre la métrica real de selección.

**Gravedad: Media.**
**Cómo arreglarlo:** una fila más en la tabla de ablación del README, con y sin `class_weight`, mismo
pipeline, mismos folds.

### R4 — MLP se queda en la ruta lenta de `KernelExplainer`

Mismo motivo (coste) que desactivó SVM/k-NN en `conf/models.yaml`, pero MLP no se reconsideró.
Benchmarkeado de forma sintética con la forma real del problema (76 features, 100 de fondo, 2000
muestras): proyecta **~10 minutos** por `/explain` en frío, uno o dos órdenes de magnitud más lento que
el resto de candidatos activos.

**Gravedad: Media.**
**Cómo arreglarlo:** medirlo y aceptarlo explícitamente (como se hizo con SVM/k-NN), o reducir
`background_samples` específicamente para MLP.

### R5 — `CalibrationConfig.method` por defecto en código es `"isotonic"` [FIXED 2026-11-22]

`src/f1cp/config/settings.py:370-374` — exactamente el método que el propio proyecto documenta como
peligroso en este dataset (isotonic sobre 3 etiquetas independientes reproduce un 100% de probabilidad de
título). Hoy enmascarado porque `conf/models.yaml` siempre lo fija a `sigmoid` explícitamente.

**Gravedad: Baja-Media** (footgun de configuración, no un bug activo).
**Cómo arreglarlo:** cambiar el default de la clase Pydantic a `"sigmoid"`.

### R6 — `pyproject.toml` declara `scikit-learn>=1.5`, el código necesita `>=1.6` [FIXED 2026-11-22]

`models/calibration.py:15` importa `sklearn.frozen.FrozenEstimator`, que no existe antes de 1.6.
`requirements.lock` fija 1.9.0, así que hoy no explota, pero el contrato declarado del paquete es
incorrecto para cualquiera que instale desde `pyproject.toml` sin el lock.

**Gravedad: Baja.**
**Cómo arreglarlo:** subir el mínimo declarado a `>=1.6`.

---

## Riesgos de infraestructura / MLOps (no re-priorizados aquí, ya detallados en la auditoría de esta sesión)

- [FIXED 2026-11-22: `tests/e2e/test_pipeline.py`] `tests/e2e/` vacío pese a documentarse como categoría activa; `pytest -m e2e` sale con éxito sobre 0
  tests recolectados — falsa sensación de cobertura.
- [FIXED 2026-11-21: sustituido por `data-sync.yml`] `nightly-update.yml` reentrena dentro de una caché de GitHub Actions efímera, sin ninguna conexión con
  el stack de Docker Compose ni con el entrypoint de Streamlit Cloud — el "despliegue" que actualiza no
  es el que sirve a nadie.
- Sin gate de cobertura (`fail_under`) en ningún sitio; `dashboard/*` y `migrations/*` excluidos del
  denominador de cobertura.
- Sin escaneo de dependencias/vulnerabilidades (sin Dependabot, sin pip-audit/trivy/bandit).
- `requirements.lock` es un `pip freeze` plano, sin hashes.
- Contenedores corren como root; sin `HEALTHCHECK` en el propio `Dockerfile` (solo en compose).
- Credenciales por defecto débiles (`f1cp`/`f1cp`) en `docker-compose.yml`/`.env.example`, puertos
  publicados al host por defecto.
- `n_jobs: -1` en los ensembles de árboles introduce no-determinismo entre máquinas, justo donde el
  top-3 de modelos ya está dentro del margen de ruido.
- `TrainingRun.error`/`IngestionReport.error` sirven `str(exc)` crudo sin autenticación de por medio
  (consecuencia directa de N5).

---

## Prioridad de corrección sugerida

| # | Qué | Estado |
|---|---|---|
| 1 | **N5** — auth mínima en `/train`, `/update-data`, `/model/{id}/activate` | **FIXED** (independiente, `feat/api-read-only-mode`) |
| 2 | **N1** — arreglar `momentum`/`_recent_rate` + test + reentrenar | **FIXED** (código + test); reentrenamiento pendiente |
| 3 | **N7** — lock en las cachés de `PredictionService` | **FIXED** |
| 4 | **N6** — validar servibilidad antes de activar un modelo | **FIXED** |
| 5 | **N4** — aviso in-sample en `home.py` | **FIXED** |
| 6 | **N3** — assert sobre `ELIMINATED_COLUMN` | **FIXED** |
| 7 | **R1** — declarar (o cerrar) la ausencia de hyperparameter tuning | Abierto — ver nota al final |
| 8 | **N2** — test dedicado de `CircuitBlock` | Abierto |
| 9 | **N8, N9** — higiene de features / `teammate_beat_rate` | **FIXED** |
| 10 | **R3, R4, R5, R6** y los riesgos de MLOps | Abierto — deuda documentada o de bajo riesgo inmediato |

**R2 ya no está abierto** — medido el 2026-08-03 (ver su entrada): la configuración actual resultó ser la
mejor de las probadas, así que se queda como está.

**Nota sobre R1 (hyperparameter tuning):** deliberadamente no abordado en esta ronda. Una búsqueda real
sobre 30 folds externos × varios candidatos multiplica el tiempo de entrenamiento por un factor grande;
la alternativa barata (una frase de limitación en el README) es la recomendada para la siguiente
iteración de documentación, no de código.

---

## Nota de alcance

Esta v2 no repite los hallazgos ya cerrados de `docs/AUDIT.md` (E1-E7, P1-P5, D1-D7) salvo para
confirmar su estado en la tabla de arriba. Para el detalle original de esos hallazgos, `docs/AUDIT.md`
sigue siendo la referencia correcta — sigue siendo precisa sobre el estado *anterior* a esta remediación,
que es exactamente para lo que sirve un documento de auditoría fechado.
