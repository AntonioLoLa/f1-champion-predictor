# Auditoría técnica — F1 Championship Predictor

**Fecha:** 2026-07-30
**Alcance:** dataset, features, validación temporal, calibración, normalización, métricas, zoo de modelos, MLflow, inferencia.
**Objeto auditado:** dataset `e6778fc17c45-2026-07-30` (19.195 × 76), training run **7** (8 candidatos, Platt), modelo activo `catboost` (id 39), modelo seleccionado `logistic_regression` (id 35).
**Estado de la suite en el momento de la auditoría:** `pytest -m "not network"` pasa (exit 0). Los defectos descritos aquí son invisibles para ella.

---

## Resumen ejecutivo

**El walk-forward está limpio. El dataset está bien construido. El problema está repartido entre la capa de features, las métricas y la inferencia — y sobre todo en que las métricas publicadas no miden lo que parecen medir.**

El número de selección de la run 7 (`logistic_regression`, `group_log_loss` 0.6039) fue reproducido de forma independiente reimplementando el camino de entrenamiento, con coincidencia a 4 decimales. Todo lo que sigue descansa sobre una lectura verificada de ese camino, no sobre inspección de código.

Lo que **no** está mal, y conviene decirlo porque se intentó romper:

- No hay contaminación temporal (30 folds enumerados; la calibración siempre es estrictamente anterior a la temporada de test).
- La etiqueta es correcta: 803 positivos, exactamente un campeón por temporada, 2026 con cero positivos y excluida por `completed_seasons`.
- No hay claves duplicadas sobre `(season, round, driver_id)`.
- Los standings son genuinamente por ronda: 0 incrementos negativos de puntos en 19.195 filas.
- Las features de carrera y títulos previos son estrictamente retrospectivas.
- La tabla de puntuación por era es correcta (9 / 10 / 25 / 26 con punto de vuelta rápida 2019–2024; sprint 3 en 2021 → 8 desde 2022; 25 de nuevo en 2025).
- `is_mathematically_eliminated` coincide con `domain/elimination.py` (22/22 vivos en 2026 R11).
- El modelo **sí** bate al baseline ingenuo "el líder actual gana": 73,0% vs 63,1% de top-1 en fase temprana, y falla exactamente en los campeonatos que fueron genuinamente impredecibles (2007, 2010, 2012, 2025).

Lo que sí está mal, ordenado por impacto en una decisión:

1. **La inferencia puntúa una población distinta de la evaluada.** El builder filtra `entered=False`; la inferencia no. Son 1.877 filas (8,9%). Demostrado: en 2020 R16 el entrenamiento vio 20 filas y la inferencia puntúa 23.
2. **La normalización por grupo fabrica respuestas seguras y falsas cuando el campeón no está en el grupo, y la métrica de selección no lo ve.** En 2020 R16 (Hamilton ausente por COVID) todas las probabilidades calibradas son ~1e-6 — el modelo dice correctamente "aquí no está el campeón" — y la normalización devuelve **Bottas 76,1%**, un piloto matemáticamente eliminado.
3. **La selección del modelo no está soportada estadísticamente.** SE del metric ≈ 0,09–0,15 frente a un rango top-3 de 0,10. LR vs LightGBM: p=0,31, y **LightGBM gana en 18 de 30 temporadas**.
4. **El dashboard muestra predicciones in-sample como si fueran rendimiento.** 2007: top-1 servido 94,1% vs OOF honesto 29,4%.
5. **Tres métricas de cabecera no aportan información:** `final_round_top1_hit_rate` es tautológica, y `precision`/`recall`/`f1` son algebraicamente el mismo número que `top1_hit_rate`.
6. **Un bloque de features entero está muerto** (99,68% nulos) por una clave de join incorrecta.

---

## Errores confirmados

### E1 — `CircuitBlock` no funciona: la clave del join es incorrecta

**Archivo:** `src/f1cp/features/circuit.py`, `CircuitBlock.build`, líneas 97–99.

```python
enriched = drivers_x_remaining.join(
    history, on=["season", "round", "driver_id", "circuit_id"], how="left"
).filter(pl.col("_prior_visits_at_circuit") >= block_config.min_history_races)
```

`history` está indexado por la ronda **en la que ese circuito se corrió**. `drivers_x_remaining` lleva `round` = la ronda *as-of* y `circuit_id` = el circuito de una ronda **futura**. El join solo casa cuando un mismo circuito se corre dos veces en la misma temporada.

**Evidencia:** `avg_finish_at_remaining_circuits` nula en 19.134 de 19.195 filas (99,68%). `remaining_circuits_visited` ∈ {0,1}, media 0,0032. Las 61 filas supervivientes son exactamente las temporadas con circuito repetido.

**Impacto:** 2 de 76 features son ruido; el bloque no aporta nada de lo que su docstring promete.
**Gravedad:** alta para la credibilidad de la capa de features, baja para las métricas (una feature muerta no cuesta nada).

### E2 — `entered` es una feature constante

**Archivos:** `src/f1cp/dataset/builder.py::_apply_filters` (elimina `entered=False`) y `build_dataset`, líneas 92–96 (la incluye en `feature_names`).

**Evidencia:** `nunique = 1` en 19.195 filas. El coeficiente de la regresión logística desplegada es exactamente `+0.0000`.

**Gravedad:** baja. Trivial de corregir.

### E3 — Train/serve skew: la inferencia no aplica los filtros del dataset

**Archivos:** `src/f1cp/dataset/builder.py::_apply_filters` frente a `src/f1cp/inference/predictor.py::_engineered_frame`, líneas 401–402 (llama a `run_pipeline` y no filtra nada).

**Evidencia:** el panel vivo tiene 21.072 filas ≥1981; el dataset entrenado tiene 19.195. Diferencia: **1.877 filas (8,9%)** con `entered=False`. En 2020 R16 el modelo se entrenó sobre 20 filas y sirve sobre 23.

**Impacto:** el denominador del `GroupProbabilityNormalizer` es distinto en evaluación y en producción. Los pilotos ausentes se llevan masa de probabilidad que en entrenamiento no existía.
**Gravedad:** alta.

### E4 — La normalización por grupo produce respuestas falsas y seguras cuando falta el campeón

**Archivos:** `src/f1cp/models/normalizer.py::GroupProbabilityNormalizer.normalize`; `src/f1cp/training/metrics.py::group_log_loss`, línea 62 (`positives = frame.filter(...)`).

**Evidencia — 2020 R16, OOF de `logistic_regression`:**

| piloto | `probability` (calibrada) | `probability_normalized` |
|---|---|---|
| Bottas | 3,0e-06 | **0,7612** |
| Verstappen | 8,7e-07 | 0,2386 |
| Pérez | 2,5e-10 | 0,0001 |

Hamilton no aparece: se ausentó del GP de Sakhir por COVID y `_apply_filters` eliminó su fila. Bottas estaba matemáticamente eliminado. **El modelo tenía razón antes de normalizar.**

`group_log_loss` filtra a filas positivas antes de calcular, así que se salta ese grupo en silencio (1 de 568). `top1_hit_rate` lo cuenta como fallo. Dos métricas discrepan sobre las mismas filas.

**Gravedad:** alta. Es el hallazgo que más pesa en la conclusión.

### E5 — Las tasas dividen por rondas de calendario, no por rondas disputadas

**Archivos:** `src/f1cp/features/rates.py::RatesBlock.build`, línea 48:

```python
rounds_completed = pl.col("round").shift(1).over(group).clip(lower_bound=1)
```

Mismo patrón en `src/f1cp/features/form.py`, línea 64 (`season_points_rate`).

**Evidencia:** Jacques Villeneuve Sr., 1981, debuta en R14 y abandona en el debut. En R15 su `dnf_rate` = **0,0714 = 1/14** en lugar de 1,0. Patrón idéntico con 1/12 en 1982.

**Alcance:** 211 de 1.308 temporadas-piloto (16%) empiezan después de la ronda 1. Todas sus tasas quedan sistemáticamente infravaloradas, y como `finish_rate = 1 − dnf_rate`, un debutante que abandonó en todas sus carreras aparece con ~0,93 de fiabilidad.

**Gravedad:** media.

### E6 — `final_round_top1_hit_rate` es tautológica

**Archivos:** `src/f1cp/training/metrics.py::final_round_top1_hit_rate`; `src/f1cp/features/season_context.py`, línea 44.

En la última ronda `max_points_remaining = 0`, luego `is_mathematically_eliminated = (gap_to_leader > 0)` es cierto para todos menos el líder — que es el campeón por definición.

**Evidencia:** 600 de 630 filas OOF de última ronda marcadas como eliminadas. El baseline "líder actual" puntúa exactamente **1,000** en esta métrica en las 30 temporadas. El 1.0 de LR en la run 7 no mide nada.

**Gravedad:** media (falsa sensación de calidad, cero impacto en el modelo).

### E7 — `accuracy`, `precision`, `recall` y `f1` no son cuatro números

**Archivo:** `src/f1cp/evaluation/metrics.py::_classification_metrics`, líneas 78–94. `y_pred` se deriva de `top_ranked_mask`, que marca exactamente un positivo por grupo. Con exactamente un positivo real por grupo, TP+FP = TP+FN = nº de grupos.

**Evidencia (OOF de LR):** precision 0,7852 · recall 0,7866 · f1 0,7859 · `top1_hit_rate` **0,7852**. Idénticos salvo por el único grupo sin campeón (E4). Y `accuracy` 0,9797 contra un clasificador constante "nunca campeón" que puntúa **0,9527**.

**Gravedad:** media (falsa sensación de calidad).

---

## Posibles errores

### P1 — La selección del ganador no está soportada por los datos

`src/f1cp/training/selection.py::select_best` usa `min()` sin ninguna noción de incertidumbre.

| modelo | GLL | SE | IC 95% |
|---|---|---|---|
| logistic_regression | 0,6039 | 0,0924 | [0,423, 0,785] |
| lightgbm | 0,6998 | 0,1482 | [0,409, 0,990] |
| random_forest | 0,7409 | 0,1489 | [0,449, 1,033] |
| knn | 1,0120 | 0,1621 | [0,694, 1,330] |
| xgboost | 1,0302 | 0,2062 | [0,626, 1,434] |
| svm | 1,0883 | 0,2195 | [0,658, 1,519] |
| catboost | 1,1765 | 0,2157 | [0,754, 1,599] |
| mlp | 1,6222 | 0,1895 | [1,251, 1,994] |

Comparaciones pareadas contra LR (30 temporadas):

| comparación | diferencia media | t pareado | Wilcoxon | LR mejor en |
|---|---|---|---|---|
| random_forest | +0,1369 | 0,101 | 0,184 | 17/30 |
| lightgbm | +0,0959 | **0,312** | **0,792** | **12/30** |
| knn | +0,4081 | <0,001 | <0,001 | 25/30 |
| xgboost | +0,4262 | 0,010 | 0,018 | 20/30 |
| svm | +0,4844 | 0,007 | 0,001 | 22/30 |
| catboost | +0,5726 | 0,002 | 0,001 | 21/30 |
| mlp | +1,0183 | <0,001 | <0,001 | 27/30 |

La conclusión honesta es "LR, LightGBM y RandomForest son indistinguibles; los otros cinco son peores", no "LR es el ganador".

### P2 — El metric de selección está dominado por unas pocas temporadas

El peor 5% de rondas concentra el 22–30% de la pérdida total según el modelo. Las peores temporadas de LR son 2010 (1,92), 2007 (1,85), 2025 (1,54) y 2012 (1,35) — campeonatos genuinamente impredecibles. Una media de 30 medias-por-fold sobre una pérdida de cola pesada no es un selector robusto: la mediana ordenaría distinto.

### P3 — Multicolinealidad severa: coeficientes sin sentido e importancias corruptas

Tres pares **exactamente** colineales:

| par | r |
|---|---|
| `finish_rate` / `dnf_rate` | −1,0000 |
| `team_reliability` / `team_dnf_rate` | −1,0000 |
| `finished` / `is_dnf` | −1,0000 |

Siete más por encima de \|0,95\|: `race_points`/`round_points` (0,995), `career_starts`/`experience_seasons` (0,973), `rounds_remaining`/`season_progress` (−0,967), `season_avg_grid`/`season_avg_quali` (0,963), `last_3_points`/`last_5_points` (0,959), `career_wins`/`prior_titles` (0,958), `current_points`/`team_points` (0,957).

Coeficientes del modelo desplegado (entradas estandarizadas):

```
-3.9160  gap_to_leader_norm
+3.3856  experience_seasons     ┐ r = 0,973 entre ellas,
-2.8606  career_starts          ┘ magnitudes que casi se cancelan
-2.5214  is_mathematically_eliminated
+2.0624  team_points
-1.9007  round_max_points
-1.8379  gap_to_leader
-1.7832  gap_to_leader_over_remaining
+1.6311  max_points_remaining
+1.3801  finish_position        ← acabar PEOR aumenta la probabilidad de título
-1.3414  prior_titles           ← haber ganado títulos la REDUCE
```

Además, la permutation importance de un par perfectamente colineal es ≈0 para **ambos** miembros, así que la página de Feature Importance los infravalora sistemáticamente.

### P4 — La calibración se mide sobre una columna que el producto nunca sirve

`training/metrics.py::compute_metrics` calcula `brier_score` y `expected_calibration_error` sobre `probability` (pre-normalización). `inference/predictor.py::_score` sirve `probability_normalized`. Ambos tie-breakers de `training.selection` son sobre la columna no servida.

**Evidencia:** suma media por grupo de la columna calibrada — LR 0,833, SVM 1,249, catboost 0,972. Los números servidos de LR son sistemáticamente un 20% más altos que aquellos cuya calibración se midió.

### P5 — El ECE reportado tiene sesgo de muestra pequeña

`trainer.py::_mean_metrics` promedia 30 ECEs calculados sobre ~400 filas cada uno. Recalculado en pool sobre las 11.982 filas OOF, el ECE de LR es **0,0080** frente al 0,0243 almacenado. Un factor 3, solo por el binning.

---

## Decisiones metodológicas discutibles

### D1 — El modelo desplegado nunca ve 2023–2025

`trainer.py::_train_one_candidate`, líneas 353–358: el refit final aplica `_split_calibration_seasons(seasons, 3)`, así que ajusta sobre 1981–2022 y solo el calibrador ve 2023–2025.

*A favor:* el calibrador es honesto. *En contra:* la era reglamentaria actual es invisible para el estimador. *Alternativa:* refit sobre todo con calibración cross-fitted.

### D2 — El dashboard presenta predicciones in-sample como rendimiento

Como el modelo servido lleva 1981–2022 en su ajuste, `/predict-season/{season}` es in-sample para casi todo el histórico:

| temporada | top-1 servido (in-sample) | top-1 OOF (honesto) | GLL servido / GLL OOF |
|---|---|---|---|
| 2007 | 94,1% | 29,4% | 0,48× |
| 2010 | 73,7% | 42,1% | 0,57× |
| 2012 | 90,0% | 50,0% | 0,58× |

Nada en la UI distingue una cosa de la otra.

### D3 — `is_mathematically_eliminated` es una feature, no una restricción

El 35,2% de las filas OOF son pilotos eliminados; reciben de media el 1,4% de la masa por ronda, y 8 filas superan el 10% de probabilidad de título. Poner a cero y renormalizar es gratis y estrictamente correcto.

### D4 — 7 de 76 features son constantes dentro del grupo de predicción

`entered`, `has_sprint`, `max_points_remaining`, `round_max_points`, `rounds_remaining`, `rounds_total`, `season_progress`.

Bajo normalización por grupo **no pueden cambiar el ranking**; en un modelo lineal solo reescalan los logits del grupo (una perilla de temperatura). Dos de ellas llevan de los mayores coeficientes del modelo (`round_max_points` −1,90, `max_points_remaining` +1,63), y `round_max_points` es un marcador puro de era (9/10/25/26).

### D5 — El test de leakage es mucho más estrecho de lo que anuncia

`tests/unit/features/test_pipeline.py::test_features_are_unchanged_by_deleting_the_future` corre sobre `tests/unit/features/synthetic.py`: 2 temporadas × 4 rondas × 3 pilotos, todos disputando todas las rondas, sin circuito repetido, sin debutantes, sin ausencias.

Demuestra genuinamente la forma del `shift(1)`. Es estructuralmente incapaz de detectar E1, E3 o E5. `CLAUDE.md` lo llama "the guard"; guarda una de varias puertas.

### D6 — El feature engineering aporta poco sobre 5 features

Reimplementando el camino de entrenamiento (reproduce 0,6039 exacto):

| features | GLL | SE | pareado vs las 76 |
|---|---|---|---|
| las 76 | 0,6039 | 0,092 | — |
| 5 (`gap_to_leader_norm`, `season_progress`, `is_mathematically_eliminated`, `points_share`, `win_rate`) | 0,6909 | 0,112 | p = **0,159** |
| 3 | 0,7592 | 0,113 | p = 0,018 |
| 2 | 0,7799 | 0,125 | p = 0,018 |

71 features compran una mejora **no significativa** de 0,087.

### D7 — `min_train_seasons: 15` ajusta realmente sobre 12

El holdout de calibración se recorta de las temporadas de entrenamiento *después* de que el splitter aplique el mínimo. El fold 0 ajusta sobre 1981–1992.

---

## Validación temporal — tabla de folds

`strategy: expanding`, `min_train_seasons: 15`, `test_horizon_seasons: 1`, `calibration_holdout_seasons: 3`.
45 temporadas completas (1981–2025); 2026 excluida por estar en curso. **30 folds. Sin contaminación temporal: la calibración es siempre estrictamente anterior al test.**

| fold | ajuste | calibración | test |
|---|---|---|---|
| 0 | 1981–1992 (12) | 1993, 1994, 1995 | 1996 |
| 1 | 1981–1993 (13) | 1994, 1995, 1996 | 1997 |
| 2 | 1981–1994 (14) | 1995, 1996, 1997 | 1998 |
| 3 | 1981–1995 (15) | 1996, 1997, 1998 | 1999 |
| 4 | 1981–1996 (16) | 1997, 1998, 1999 | 2000 |
| 5 | 1981–1997 (17) | 1998, 1999, 2000 | 2001 |
| 6 | 1981–1998 (18) | 1999, 2000, 2001 | 2002 |
| 7 | 1981–1999 (19) | 2000, 2001, 2002 | 2003 |
| 8 | 1981–2000 (20) | 2001, 2002, 2003 | 2004 |
| 9 | 1981–2001 (21) | 2002, 2003, 2004 | 2005 |
| 10 | 1981–2002 (22) | 2003, 2004, 2005 | 2006 |
| 11 | 1981–2003 (23) | 2004, 2005, 2006 | 2007 |
| 12 | 1981–2004 (24) | 2005, 2006, 2007 | 2008 |
| 13 | 1981–2005 (25) | 2006, 2007, 2008 | 2009 |
| 14 | 1981–2006 (26) | 2007, 2008, 2009 | 2010 |
| 15 | 1981–2007 (27) | 2008, 2009, 2010 | 2011 |
| 16 | 1981–2008 (28) | 2009, 2010, 2011 | 2012 |
| 17 | 1981–2009 (29) | 2010, 2011, 2012 | 2013 |
| 18 | 1981–2010 (30) | 2011, 2012, 2013 | 2014 |
| 19 | 1981–2011 (31) | 2012, 2013, 2014 | 2015 |
| 20 | 1981–2012 (32) | 2013, 2014, 2015 | 2016 |
| 21 | 1981–2013 (33) | 2014, 2015, 2016 | 2017 |
| 22 | 1981–2014 (34) | 2015, 2016, 2017 | 2018 |
| 23 | 1981–2015 (35) | 2016, 2017, 2018 | 2019 |
| 24 | 1981–2016 (36) | 2017, 2018, 2019 | 2020 |
| 25 | 1981–2017 (37) | 2018, 2019, 2020 | 2021 |
| 26 | 1981–2018 (38) | 2019, 2020, 2021 | 2022 |
| 27 | 1981–2019 (39) | 2020, 2021, 2022 | 2023 |
| 28 | 1981–2020 (40) | 2021, 2022, 2023 | 2024 |
| 29 | 1981–2021 (41) | 2022, 2023, 2024 | 2025 |
| **refit final (desplegado)** | **1981–2022 (42)** | **2023, 2024, 2025** | — |

---

## Riesgos futuros

- **`SimpleImputer` descarta columnas todo-NaN en tiempo de fit.** Con `avg_finish_at_remaining_circuits` al 99,7% de nulos, un fold temprano puede quedarse sin la columna mientras otro la conserva: la anchura efectiva de la matriz varía por fold en silencio (`UserWarning: Skipping features without any observed values` al reejecutar los folds).
- **Cada `train run` añade candidatos.** `model_versions` va por 42 filas y 7 runs. Es por diseño, pero `is_best` e `is_active` divergiendo (35 vs 39 hoy) sin nada que lo señale es una trampa operativa.
- **La normalización por grupo es frágil ante grupos degenerados.** Con probabilidades calibradas del orden de 1e-10, dividir por la suma del grupo amplifica ruido de coma flotante hasta el 76% de confianza (E4). En cuanto un piloto no clasificado o un grupo raro entre en el panel de inferencia, esto reaparece.
- **El `random_state` está fijo pero la varianza entre semillas no se mide.** Reentrenar con otra semilla puede cambiar el ganador declarado, y nada lo detectaría.
- **La feature `trend` ya cambió una vez de forma silenciosa** (truncamiento de dtype en `rolling_map`). El mismo patrón — una expresión polars devolviendo el dtype de entrada — sigue en `cum_mean`, usado por `qualifying`, `circuit` y `constructor`; merece una comprobación de dtype como la que se le puso a `trend`.

---

## Prioridad de corrección

| # | Problema | Por qué en esta posición |
|---|---|---|
| 1 | **E3 + E4** — paridad de filtros en inferencia y guarda en la normalización | Produce respuestas servidas incorrectas *hoy* |
| 2 | **D2** — marcar o sustituir las predicciones históricas in-sample | Es lo que más sobreestima el producto de cara al exterior |
| 3 | **P1** — selección con incertidumbre (comparación pareada, no `min()`) | La afirmación central del proyecto no está soportada |
| 4 | **E6 + E7** — retirar o reetiquetar las métricas tautológicas; regenerar el README | El README además está desactualizado (run 3, seis candidatos, pre-Platt) |
| 5 | **E1** — arreglar o eliminar `CircuitBlock` | Un bloque entero que no hace nada |
| 6 | **E5** — denominadores de las tasas | 16% de las temporadas-piloto con features sesgadas |
| 7 | **P4 + P5** — medir calibración sobre la columna servida; ECE agrupado | Las cifras de calibración no describen el producto |
| 8 | **P3 + D4** — podar colineales y constantes-por-grupo | Estabilidad e interpretabilidad |
| 9 | **E2** — quitar `entered` | Trivial |
| 10 | **D1, D5, D7** — estrategia de refit, ampliar el fixture de leakage, semántica de `min_train_seasons` | Deuda estructural |

---

## Evaluación del problema: ¿es realmente abordable?

Sí, pero es más pequeño de lo que el tamaño del dataset sugiere. 45 temporadas = 45 etiquetas independientes; 30 de test. Todo lo que se afirme a nivel de temporada tiene n=30.

**Qué sería razonable esperar:** top-1 temprano en la banda 60–75% (el baseline "líder actual" da 63,1%), tardío casi determinista, y un `group_log_loss` agrupado de ~0,6–0,9 con ~20 pilotos por grupo (el uniforme es log 20 = 3,0).

**Baseline ingenuo medido — "el líder del campeonato es el campeón":**

| fase | baseline | LR (OOF) |
|---|---|---|
| temprana | 63,1% | 73,0% |
| media | 69,2% | 79,5% |
| tardía | 84,6% | 85,3% |
| agrupado | 72,7% | 79,4% |
| última ronda | **100%** | **100%** |

**¿Son creíbles las métricas?** Como valores out-of-fold, **sí**. El modelo bate al baseline ingenuo donde importa (fase temprana) y falla donde el deporte fue impredecible. **No** son creíbles como evidencia de un *ranking* entre familias de modelos, y las del README están obsoletas.

**El planteamiento no está roto.** Lo que está roto es la traducción de "puntuación por grupo" a "distribución de probabilidad" cuando el grupo no contiene al campeón, y la presentación.

---

## Valoración general

| Área | Nota | Razón |
|---|---|---|
| Calidad del dataset | **7/10** | Claves, etiqueta y standings verificados correctos; dos features muertas y un bug de denominador |
| Feature engineering | **4/10** | Un bloque no funcional, una constante, tres pares exactamente colineales, 7 constantes por grupo, y 71 de 76 features sin aporte significativo |
| Validación temporal | **8/10** | Walk-forward genuinamente limpio; pierde por el holdout recortado del mínimo y por la agregación media-de-folds |
| Entrenamiento | **7/10** | Orquestación correcta y reproducible (0,6039 reproducido exacto); `class_weight="balanced"` + calibración + normalización son tres correcciones solapadas |
| Calibración | **6/10** | Platt sobre 3 temporadas es la decisión correcta y está bien argumentada; medida sobre una columna que no se sirve y con sesgo 3× |
| Inferencia | **4/10** | Reconstruye features correctamente y la invalidación de caché es sólida, pero puntúa una población distinta de la evaluada |
| Arquitectura ML | **9/10** | Genuinamente buena: composition root, un solo pipeline para train y serve, datasets versionados, artefactos OOF, `is_best`/`is_active` separados |
| Robustez estadística | **3/10** | La decisión de selección no está soportada; tres métricas tautológicas o derivadas reportadas como evidencia independiente |

---

## Conclusión

**¿Confiaría en este pipeline para publicar sus resultados en producción? No en su estado actual — pero por razones más acotadas de lo que parece.**

Conviene separar cuatro cosas:

1. **Las probabilidades de la ronda viva actual (2026 R11): sí, publicables.** Son out-of-sample, coherentes con la lógica de eliminación (22/22 vivos, feature y módulo de dominio de acuerdo), y no hay ausentes en ese grupo. Con la salvedad de E3 en cuanto haya uno.

2. **Las páginas de "season replay" histórico: no.** Son in-sample y sobreestiman el rendimiento en torno a 2× precisamente en las temporadas que un evaluador técnico miraría primero (2007: 94,1% vs 29,4%).

3. **El leaderboard de modelos y la afirmación "seleccionado entre 8 candidatos": no.** LightGBM gana en 18 de 30 temporadas frente al ganador declarado, con p=0,31. La afirmación no es defendible bajo preguntas.

4. **Las tablas del README: no.** Están obsoletas (run 3, seis candidatos, pre-Platt) y descansan sobre `final_round_top1_hit_rate` — que un baseline trivial puntúa 1,000 — y sobre `accuracy` 97,5% contra un 95,3% de un clasificador constante.

La evidencia que más pesa en este juicio es **E4**: en 2020 R16 el modelo produjo la respuesta correcta (~1e-6 para todos, "el campeón no está aquí") y **la capa de normalización la convirtió en un 76% de confianza sobre un piloto matemáticamente eliminado**, sin que ninguna métrica del proyecto lo registrara. Un pipeline en el que la capa de post-proceso puede invertir una respuesta correcta y el sistema de evaluación no se entere no está listo para publicar.

Es también, afortunadamente, el arreglo más barato de la lista.
