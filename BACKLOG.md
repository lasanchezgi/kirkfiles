# Backlog — The Kirk Files

> **C.2 — Clasificador de tipo de cambio narrativo:** ✅ **Completa** (ver README).
> Implementada en `pipeline/analyzer.py` (`run_c2`), prompt en
> `prompts/classify_change_type.yaml`, tests 20/20 en `tests/test_analyzer.py`. Las 16
> contradicciones `direct` limpias ya están clasificadas en DB: 11 silent · 5
> evidence_based · 0 acknowledged. No queda trabajo pendiente.

## E.2 — Vista: Coherencia (scorecard narrativo) ✅ UI implementada

> **Estado (2026-06-21):** La vista `📊 Coherencia` está construida en `ui/app.py`
> (`view_coherence` + `scorecard()`), registrada en el NAV y probada con `AppTest`.
> Las 4 dimensiones calculan en vivo desde la DB (31% / 39% / 31% / 0%, idénticos a
> los valores proyectados abajo). El banner de advertencia de "scorecard parcial" se
> muestra mientras `verifications < 192` (Nivel 2). **Lo único pendiente es dato, no
> código.**
>
> **Único paso restante — correr el Nivel 2:**
>
> ```bash
> python -m pipeline.verifier --level 2
> ```
>
> Esto verifica todos los claims con fuente citada para tener cobertura de todos los
> episodios (~192 claims, ~$15). Al terminar, la dimensión de *respaldo externo* deja
> de ser parcial, el banner de advertencia desaparece y el scorecard se recalcula solo
> (no hay que tocar código). Antes de gastar, se puede estimar con
> `python -m pipeline.verifier --level 2 --dry-run`.

**Contexto:**
El dashboard muestra los datos desagregados (contradicciones, verificaciones, claims)
pero no responde la pregunta central del proyecto:
*¿La investigación de Candace Owens es internamente coherente y externamente respaldada?*

Esta vista no emite un veredicto — presenta un scorecard por dimensión para que
el usuario conecte los puntos. Fiel al principio: "dejar servida la información."

**Bloqueo:** Requiere Nivel 2 de verificaciones (Fase D completa) para que las
métricas de respaldo externo sean representativas. Con solo 28 claims verificados
(Nivel 1), el scorecard sería parcial y potencialmente engañoso.

**4 dimensiones del scorecard:**

| Dimensión | Qué mide | Fuente de datos |
|---|---|---|
| Consistencia interna | % contradicciones direct con change_type = silent vs evidence_based | tabla contradictions |
| Respaldo externo | % claims verificados con verdict = supported vs contradicted | tabla verifications |
| Evolución con evidencia | % cambios de posición respaldados por evidencia intermedia | contradictions.change_type |
| Transparencia narrativa | % cambios reconocidos explícitamente (acknowledged) | contradictions.change_type |

**UI:**
- 4 barras de progreso horizontales con label y porcentaje
- Sin score agregado, sin veredicto final
- Nota al pie: "Basado en X claims verificados de Y totales — scorecard parcial hasta
  completar Nivel 2 de verificación"
- Si Nivel 2 no está completo: banner de advertencia visible antes del scorecard

**Valores actuales (Nivel 1 — parciales):**
- Consistencia interna: 5/16 evidence_based = 31% con sustento
- Respaldo externo: 11/28 supported = 39% (solo Nivel 1)
- Evolución con evidencia: 5/16 = 31%
- Transparencia narrativa: 0/16 = 0% reconocidos

**Prioridad:** Media — depende de Nivel 2 ($15 USD, ~192 claims)
**Costo implementación:** $0.00 — solo UI sobre datos existentes
**Desbloquea:** La pregunta central del proyecto de forma defendible
