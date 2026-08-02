# Plan de mejora — triaje de la auditoría

**Entrada:** `docs/AUDIT.md` (2026-07-30).
**Criterio:** simplicidad, robustez, rigor estadístico. Se descarta todo lo que añada complejidad estructural para ganar poco.
**Resultado:** 11 decisiones. 4 críticas, 4 importantes, 3 opcionales, 5 explícitamente descartadas.

---

## Tabla de decisiones

| # | Cambio | Origen | Cat. | Grav. | Dificultad | Tiempo | Riesgo | ¿Merece la pena? |
|---|---|---|---|---|---|---|---|---|
| 1 | Unificar la población puntuada | E3, E4 | **A** | Alta | Baja | 3–4 h + retrain | Medio | **Sí. Es el fix del proyecto.** |
| 2 | Podar el zoo: fuera SVM y k-NN | — | **A** | Media | Trivial | 30 min | Nulo | **Sí, y primero** (habilita iterar) |
| 3 | Selección con incertidumbre (regla 1-SE) | P1, P2 | **A** | Alta | Baja | 2 h | Bajo | **Sí. Es la afirmación central.** |
| 4 | Marcar in-sample vs out-of-sample | D2 | **A** | Alta | Baja | 2–3 h | Bajo | **Sí. Agujero de credibilidad.** |
| 5 | Sanear la superficie de métricas | E6, E7, P4, P5 | **A** | Media | Baja | 3 h | Bajo | **Sí.** |
| 6 | Eliminación matemática como restricción | D3 | **B** | Media | Baja | 2 h | Bajo | Sí |
| 7 | Denominadores de tasas | E5 | **B** | Media | Trivial | 1 h | Nulo | Sí |
| 8 | Higiene de features (duplicados exactos) | P3, E2 | **B** | Baja | Trivial | 30 min | Nulo | Sí |
| 9 | Ampliar el fixture de leakage | D5 | **B** | Media | Baja | 2 h | Nulo | Sí |
| 10 | `CircuitBlock`: arreglar, medir, decidir | E1 | **C** | Baja | Media | 3 h | Bajo | Solo si sobra tiempo |
| 11 | Tabla de ablación en el README | D6 | **C** | Baja | Trivial | 1 h | Nulo | Sí, si se publica el proyecto |
| — | Features constantes por grupo | D4 | **D** | — | — | — | — | **No tocar** |
| — | Refit sin 2023–2025 | D1 | **D** | — | — | — | — | **No tocar** |
| — | Grano del panel / normalización | E4 | **D** | — | — | — | — | **No tocar** |
| — | `min_train_seasons` | D7 | **D** | — | — | — | — | Un comentario, no un cambio |
| — | Arquitectura (container, layering, HTTP) | — | **D** | — | — | — | — | **No tocar** |

---

## A. Errores críticos

### 1. Unificar la población puntuada

**Origen:** E3 (train/serve skew, 1.877 filas) + E4 (Bottas 76% en 2020 R16).

**Qué hacer:** poner `exclude_non_starters: false` en `conf/features.yaml` y borrar la rama correspondiente de `dataset/builder.py::_apply_filters`.

**Por qué esa dirección y no la contraria.** La alternativa —aplicar el filtro también en inferencia— es igual de barata y está **mal**: excluiría a Hamilton de la predicción de 2020 R16 cuando lideraba el campeonato por 127 puntos. La pregunta del producto es "después de la ronda R, quién gana el título", y el conjunto de respuestas es *quien está en el campeonato*, no *quien corrió esa carrera*. Faltar a un Gran Premio no te saca del mundial.

**Por qué es más barato de lo que parece.** El panel ya es denso *antes* del filtro, así que las ventanas móviles de las demás filas ya incluyen la ronda saltada como un cero. Quitar el filtro solo cambia (a) qué filas son ejemplos de entrenamiento y (b) qué filas entran en el grupo de normalización. **Ningún valor de feature de ninguna otra fila cambia.**

**Añadir además:** que `group_log_loss` cuente y registre los grupos sin fila positiva en lugar de saltárselos en silencio, y un assert en el trainer de que ese contador es 0.

| | |
|---|---|
| **Coste** | 3–4 h + reentrenamiento |
| **Beneficio** | Evaluación e inferencia puntúan la misma población. Desaparece el caso 2020 R16. `group_log_loss` deja de tener un punto ciego. |
| **Impacto en métricas** | 19.195 → 21.072 filas, 568 → ~600 grupos. Dirección desconocida, magnitud pequeña. Los números publicados **cambian** y hay que regenerarlos. |
| **Impacto en mantenibilidad** | Positivo: desaparece una divergencia entre dos módulos que nada obligaba a mantener sincronizados. |
| **Riesgo** | Medio: invalida todos los modelos registrados. Hay que reentrenar y regenerar el README. |

**Por qué no hacerlo:** ninguna razón que resista. Es el defecto que hace que las probabilidades servidas no sean las evaluadas.

---

### 2. Podar el zoo: fuera SVM y k-NN

**Qué hacer:** `enabled: false` para `svm` y `knn` en `conf/models.yaml`.

**Por qué va el primero.** No es cosmética, es un habilitador. La run 7 tardó **2 h 19 min** de reloj. `SVC(probability=True)` ajusta un calibrador Platt interno de 5 folds sobre ~19k filas en cada uno de los 30 folds externos, y es con diferencia el candidato más lento. k-NN es el único sin explainer SHAP nativo: los ~18 s de `/explain` en frío son **enteramente** suyo. Con los dos fuera, el ciclo de iteración baja a ~40 min y la peor latencia de la app desaparece.

**Coste de oportunidad:** ninguno medido. Ambos están significativamente por detrás del ganador (p=0,0069 y p<0,001) y ninguno de los dos es una familia que un revisor eche de menos junto a LR + RF + 3 GBM + MLP.

| | |
|---|---|
| **Coste** | 30 min |
| **Beneficio** | Entrenamiento 3,5× más rápido; se elimina la peor latencia del producto |
| **Impacto en métricas** | Ninguno |
| **Impacto en mantenibilidad** | Positivo: dos ramas menos en `models/base.py` y en el dispatch de SHAP |

**Por qué no hacerlo:** si el objetivo fuera enseñar amplitud de catálogo. No lo es — 6 familias ya son amplitud, y un zoo que tarda 2 h desincentiva reentrenar, que es peor.

---

### 3. Selección con incertidumbre: regla de 1 error estándar

**Origen:** P1 (LightGBM gana en 18 de 30 temporadas frente al ganador declarado, p=0,31) + P2.

**Qué hacer:** `training/selection.py::select_best` ya recibe `fold_metrics` a través de `CandidateResult`. Calcular el SE de la media entre folds y aplicar la regla clásica: **entre los candidatos cuya media cae dentro de 1 SE de la mejor, elegir el primero en un orden de preferencia configurado** (más simple primero). Registrar el grupo empatado en el `TrainingRun` y mostrarlo en el leaderboard.

**Qué pasa con los datos actuales.** El mejor es 0,6039 con SE 0,0924; el umbral queda en 0,6963 y LightGBM está en 0,6998 — justo fuera. **La regla no cambia el ganador, lo justifica.** Y la tabla pasa de "LR gana" a "LR, LightGBM y RandomForest son indistinguibles; se elige LR por parsimonia" — que es lo que dicen los datos.

| | |
|---|---|
| **Coste** | 2 h, ~40 líneas, sin dependencias nuevas |
| **Beneficio** | La afirmación central del proyecto pasa a estar soportada |
| **Impacto en métricas** | Ninguno |
| **Impacto en mantenibilidad** | Neutro |

**Por qué no hacerlo:** si se prefiriera un test bayesiano o un Nadeau-Bengio corregido. Sería sobreingeniería: con n=30 folds y una decisión binaria, 1-SE es suficiente, es estándar (Hastie & Tibshirani) y se explica en una frase.

---

### 4. Distinguir predicciones in-sample de out-of-sample

**Origen:** D2 (2007: top-1 servido 94,1% vs OOF honesto 29,4%).

**Qué hacer:**
1. Persistir `train_seasons` y `calibration_seasons` en `ModelVersion` (ya se calculan en `_split_calibration_seasons`; caben en la columna `params` existente).
2. Añadir `is_out_of_sample: bool` a la respuesta de `/predict` y `/predict-season`.
3. En el dashboard, un aviso claro en las páginas históricas.

**Por qué esto y no servir el OOF.** Servir las predicciones OOF almacenadas para temporadas completadas sería la respuesta *idealmente* correcta, pero abre un segundo camino de datos en `PredictionService` (leer parquet vs puntuar el modelo) con su propia gestión de errores cuando el parquet no existe. Etiquetar cuesta un cuarto y es igual de honesto.

**El ángulo de portfolio.** Haber detectado esto y haberlo etiquetado es más señal de criterio que no tener el problema. Un revisor senior que vea "estas cifras son in-sample; la cifra honesta está en la página de rendimiento" concluye que sabes lo que estás mirando.

| | |
|---|---|
| **Coste** | 2–3 h |
| **Beneficio** | Se cierra el mayor agujero de credibilidad del producto |
| **Impacto en métricas** | Ninguno; cambia lo que las cifras *afirman* |
| **Impacto en mantenibilidad** | Ligeramente positivo: el modelo pasa a saber sobre qué se ajustó |

**Por qué no hacerlo:** ninguna razón. Sin esto, la demo enseña 94% donde la verdad es 29%.

---

### 5. Sanear la superficie de métricas

**Origen:** E6 (final-round tautológica), E7 (precision = recall = f1 = top1), P4 (calibración medida sobre columna no servida), P5 (ECE con sesgo 3×).

**Qué hacer, en un solo cambio:**
- Eliminar `precision`, `recall` y `f1` de `evaluation/metrics.py`. Son el mismo número que `top1_hit_rate` por construcción; publicarlos tres veces es ruido.
- Añadir una columna de **baseline** junto a `top1_hit_rate` y `final_round_top1_hit_rate`: el heurístico "el líder actual gana" (63,1% / 69,2% / 84,6% / 72,7% / **100%**). No borrar la métrica de última ronda: ponerle al lado el 100% del baseline es más instructivo que quitarla, y explica por sí solo por qué es tautológica.
- Calcular Brier y ECE **también** sobre la columna normalizada (la que se sirve), no solo sobre la calibrada.
- Calcular el ECE agrupando todas las filas OOF en lugar de promediar 30 ECEs de ~400 filas (0,0243 → 0,0080 solo por eso).

**Toda métrica junto a su baseline trivial** es la regla; es lo que convierte una tabla de resultados en un argumento.

| | |
|---|---|
| **Coste** | 3 h |
| **Beneficio** | Todo número publicado pasa a ser defendible bajo preguntas |
| **Impacto en métricas** | El ECE publicado baja ~3×; `accuracy` desaparece o va junto a su 95,3% |
| **Impacto en mantenibilidad** | Positivo: tres métricas menos que mantener y explicar |

---

## B. Mejoras importantes

### 6. Eliminación matemática como restricción, no como feature

**Origen:** D3 (35,2% de las filas son pilotos eliminados; se llevan el 1,4% de la masa por ronda; 8 filas superan el 10%).

**Qué hacer:** poner la probabilidad a 0 para `is_mathematically_eliminated` **antes** de normalizar, en la misma función compartida por trainer e inferencia. Cinco líneas.

**Por qué es buen ML y no un truco.** Es conocimiento de dominio expresado como restricción dura donde la restricción es exacta, no aprendida. `domain/elimination.py` ya existe, ya está verificado y ya coincide con la feature (22/22 vivos en 2026 R11). Un modelo probabilístico que asigna masa positiva a un evento imposible está mal, punto.

**Crítico:** aplicarlo en entrenamiento **y** en inferencia. Aplicarlo solo en inferencia reintroduce exactamente el sesgo que arregla el punto 1.

| | |
|---|---|
| **Coste** | 2 h |
| **Beneficio** | El producto deja de poder mostrar odds de título a un piloto eliminado |
| **Impacto en métricas** | `group_log_loss` mejora: el 1,4% de masa desperdiciada vuelve a pilotos vivos, y el campeón nunca está eliminado. Mejora pequeña pero **gratis** y concentrada en la fase tardía |
| **Impacto en mantenibilidad** | Neutro |

---

### 7. Denominadores de tasas

**Origen:** E5 (Villeneuve Sr. 1981: `dnf_rate` 0,0714 en lugar de 1,0; 16% de temporadas-piloto).

**Qué hacer:** en `features/rates.py` línea 48 y `features/form.py` línea 64, sustituir `pl.col("round").shift(1)` por el recuento acumulado de rondas realmente disputadas dentro del grupo.

| | |
|---|---|
| **Coste** | 1 h con test |
| **Beneficio** | Corrección |
| **Impacto en métricas** | Despreciable — ningún campeón debuta a mitad de temporada |
| **Impacto en mantenibilidad** | Neutro |

**Honestidad sobre este punto:** es un bug real que no afecta a los resultados. Se arregla porque es barato y porque un revisor que mire `rates.py` lo verá.

---

### 8. Higiene de features

**Origen:** P3 (tres pares con r = ±1,0000) + E2 (`entered` constante).

**Qué hacer:** eliminar `finish_rate`, `team_reliability`, `finished` y `entered`. Los tres primeros son `1 − x` de otra columna presente; el cuarto tiene coeficiente exactamente `+0.0000`.

**Lo que arregla de verdad:** la permutation importance de un par perfectamente colineal es ≈0 para **ambos** miembros. La página de Feature Importance miente hoy sobre cuatro features. Quitar los duplicados hace que empiece a decir la verdad.

| | |
|---|---|
| **Coste** | 30 min + reentrenamiento (combinar con el punto 1) |
| **Beneficio** | 76 → 72 features, cero información perdida, importancias interpretables |
| **Impacto en métricas** | Ninguno (información idéntica) |
| **Impacto en mantenibilidad** | Positivo |

**Qué NO hacer aquí:** no atacar los pares con r ≈ 0,95–0,97 (`career_starts`/`experience_seasons`, `career_wins`/`prior_titles`). No son duplicados; llevan información distinta aunque correlacionada, y podarlos a ojo es empezar a hacer selección de features manual, que es justo lo que la regularización L2 ya hace y hace mejor.

---

### 9. Ampliar el fixture de leakage

**Origen:** D5 (2 temporadas × 4 rondas × 3 pilotos, todos disputando todo).

**Qué hacer:** añadir a `tests/unit/features/synthetic.py` un debutante a mitad de temporada, una ausencia de una ronda y un circuito repetido dentro de la misma temporada.

**Por qué importa.** El proyecto se vende como "leakage-audited feature layer". El fixture actual es estructuralmente incapaz de detectar tres de los cinco bugs de features que encontró la auditoría. Con esos tres casos añadidos, el test habría cazado E1 y E5 solo.

| | |
|---|---|
| **Coste** | 2 h |
| **Beneficio** | La afirmación de la portada pasa a ser cierta |
| **Impacto en métricas** | Ninguno |
| **Impacto en mantenibilidad** | Alto: es la red que evita repetir esta auditoría |

---

## C. Mejoras opcionales

### 10. `CircuitBlock`: arreglar, medir, decidir

**Origen:** E1 (99,68% nulos por clave de join incorrecta).

**Qué hacer:** rehacer `history` como una búsqueda as-of por `(driver_id, circuit_id)` en lugar de un join por `(season, round, driver_id, circuit_id)`. Medir el delta de `group_log_loss`. **Si no aporta, borrar el bloque y dejar escrito que se midió.**

**Por qué el orden importa.** Borrar sin arreglar es descartar una idea que nunca se probó. Arreglar y luego borrar es una decisión de ingeniería documentada. El coste de la diferencia son 3 h, y D6 predice que el resultado será "no aporta" — pero entonces se puede *decir*.

| | |
|---|---|
| **Coste** | 3 h |
| **Beneficio** | Un bug menos, o una decisión documentada |
| **Impacto en métricas** | Casi nulo (según D6) |
| **Impacto en mantenibilidad** | Positivo en cualquiera de los dos desenlaces |

**Por qué podría no hacerse:** es el único punto de la lista cuyo beneficio esperado en métricas se sabe de antemano que es ~0.

---

### 11. Tabla de ablación en el README

**Origen:** D6 (5 features llegan a 0,6909 vs 0,6039 con 76; p=0,159).

**Qué hacer:** publicar la tabla. El análisis ya está hecho en `docs/AUDIT.md`.

**Por qué es el punto con mejor relación señal/coste de toda la lista.** Casi ningún proyecto de portfolio mide su propio feature engineering, y prácticamente ninguno publica que la mayor parte no paga. Es la diferencia entre "construí 76 features" y "construí 76 features, medí cuáles importan, y estas cinco explican casi todo". La segunda frase la firma un senior.

| | |
|---|---|
| **Coste** | 1 h |
| **Beneficio** | Alto en credibilidad |
| **Impacto en métricas** | Ninguno |

---

## D. No tocar

### Features constantes dentro del grupo (D4)

`has_sprint`, `max_points_remaining`, `round_max_points`, `rounds_remaining`, `rounds_total`, `season_progress` no pueden cambiar el ranking bajo normalización por grupo — pero **no son inútiles**: desplazan el logit de todo el grupo por igual y, a través de la no linealidad del sigmoide, controlan cuán concentrada queda la distribución. Son un control de temperatura. Quitarlas degradaría la calibración sin mejorar nada.

**Acción:** documentarlo en el docstring de `models/normalizer.py`. Es una observación no obvia y explicarla vale más que cualquier cambio de código. (`entered` sí se va — es constante *global*, no solo dentro del grupo.)

### Refit sin 2023–2025 (D1)

Arreglarlo bien exige calibración cross-fitted: K-fold sobre temporadas, calibrador ajustado sobre las predicciones out-of-fold, base reajustada sobre todo. Es más maquinaria en el trainer para recuperar 3 de 45 temporadas en el ajuste, a cambio de un calibrador algo menos limpio.

**Mala relación coste/beneficio.** El trade-off actual —perder 3 temporadas para que el calibrador sea honesto— es la decisión correcta. **Acción:** una frase en el docstring de `Trainer` explicando por qué. Un trade-off documentado es mejor ingeniería que un trade-off eliminado a base de complejidad.

### El grano del panel y la normalización por grupo

Están bien. El caso 2020 R16 no era un fallo de la normalización, era un fallo de *qué filas entraban al grupo*, y lo arregla el punto 1. Meter salvaguardas dentro del `GroupProbabilityNormalizer` sería tratar el síntoma y dejar la causa.

### `min_train_seasons` (D7)

Que `15` produzca un ajuste sobre 12 es sorprendente, pero renombrar el parámetro o cambiar la semántica invalida la configuración existente por un problema puramente de nomenclatura. **Un comentario en `conf/models.yaml`.**

### La arquitectura

`container.py`, la regla de capas, el dashboard hablando solo HTTP, los datasets versionados, los artefactos OOF, `is_best` separado de `is_active`. Es la mejor parte del proyecto (9/10 en la auditoría). No tocar nada de esto.

---

## Hoja de ruta

### Fase 1 — Sin esto el proyecto tiene defectos

**~12 h + 1 reentrenamiento**

| orden | qué | por qué en este orden |
|---|---|---|
| 0 | **Primer commit.** Todo está sin versionar en `main` | No hay punto de retorno para nada de lo que sigue |
| 1 | Punto 2 — podar SVM y k-NN | Baja el ciclo de 2 h 19 min a ~40 min. Todo lo demás se prueba más rápido |
| 2 | Punto 1 — unificar la población | Cambia el dataset; todo lo posterior depende de él |
| 3 | Punto 8 — higiene de features | Mismo reentrenamiento que el anterior |
| 4 | Punto 5 — sanear métricas | Antes de reentrenar, para no medir dos veces |
| 5 | Punto 3 — selección 1-SE | — |
| 6 | Punto 4 — flag in-sample | — |
| 7 | `dataset build` + `train run` + `evaluate` + **regenerar el README** | El README actual reporta la run 3 |

**Estado al terminar:** todo número publicado es honesto y reproducible desde `make pipeline`.

### Fase 2 — Proyecto sólido

**~5 h**

- Punto 6 — eliminación como restricción
- Punto 7 — denominadores de tasas
- Punto 9 — fixture de leakage
- Actualizar los docstrings de D4 y D1 con los trade-offs

**Estado al terminar:** el producto es lógicamente consistente y los tests cubren las clases de bug que la auditoría encontró a mano.

### Fase 3 — Proyecto excelente

**~4 h**

- Punto 11 — tabla de ablación en el README
- Punto 10 — `CircuitBlock`: arreglar, medir, decidir
- Una sección "Limitaciones" en el README: n=30 temporadas, tres modelos indistinguibles, el techo de las features

**Estado al terminar:** el proyecto argumenta sus propios límites. Ahí es donde deja de parecer un ejercicio.

### Fase 4 — Solo si sobra tiempo

**Nada.**

Es una respuesta deliberada. Después de la Fase 3 el rendimiento marginal de tocar más código es negativo: cada cambio adicional arriesga la simplicidad que hace legible el proyecto. Si sobra tiempo, se dedica a escribir mejor el README y a preparar cómo se defiende, no a añadir features.

---

## Respuesta honesta

### Qué haría de verdad

Los puntos 1 a 5, en un fin de semana, y las Fases 2 y 3 en otro. Doce horas de trabajo real convierten el proyecto de "tiene defectos que un revisor encontrará" en "un revisor encuentra que ya los encontraste tú".

El orden importa más que la lista: **podar el zoo primero**. Un ciclo de 2 h 19 min hace que no reentrenes, y no reentrenar es lo que dejó el README describiendo la run 3 con seis candidatos.

### Qué ignoraría

- **La calibración cross-fitted (D1).** Maquinaria a cambio de tres temporadas.
- **Podar las correlaciones de 0,95.** Es selección manual de features; para eso está la regularización.
- **Convertir la comparación de modelos en algo más sofisticado que 1-SE.** Con n=30 y una decisión binaria, cualquier cosa más elaborada es decoración.
- **Servir OOF para temporadas históricas.** Etiquetar cuesta un cuarto y es igual de honesto.

### Qué eliminaría completamente

- **SVM y k-NN.** El 70% del tiempo de entrenamiento y el 100% de la peor latencia de la app, a cambio de dos candidatos significativamente peores.
- **`precision`, `recall`, `f1`.** Son el mismo número que `top1_hit_rate`.
- **`finish_rate`, `team_reliability`, `finished`, `entered`.** Cuatro columnas con cero información.
- **El bloque `modern` si sigue desactivado tras la Fase 3.** Un bloque de features que nunca se ha ejecutado es código muerto con proveedores detrás. Vale más una línea en el README diciendo que se consideró y por qué no está, que un `enabled: false` que nadie ha probado nunca.

### Cuándo está terminado

Cuando se cumplan tres cosas:

1. `make pipeline` reproduce, de cero, todos los números del README.
2. Toda métrica publicada aparece junto a su baseline trivial.
3. El repositorio tiene un historial de commits que cuenta cómo se llegó hasta ahí.

**No cuando las métricas dejen de mejorar.** Este problema tiene 45 etiquetas independientes y un techo bajo — la auditoría ya midió que 71 de las 76 features no compran nada significativo. Seguir buscando décimas es la forma equivocada de gastar el tiempo.

Lo que hace serio a este proyecto no es su `group_log_loss`. Es que sepa exactamente por qué vale 0,60 y no 0,40, y que lo diga.
