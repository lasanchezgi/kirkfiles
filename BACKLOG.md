# Backlog — The Kirk Files

## C.2 — Clasificador de tipo de cambio narrativo

**Contexto:**
El sistema detecta *que* una posición cambió entre episodios, pero no *cómo* cambió.
Un cambio silencioso (inconsistencia real) no es lo mismo que un cambio construido
sobre evidencia presentada en episodios intermedios (investigación funcionando).

**Campo a agregar en tabla `contradictions`:**
`change_type TEXT` con valores:

- `silent` — la posición cambió sin reconocimiento ni evidencia en episodios intermedios
- `acknowledged` — Candace reconoció explícitamente el cambio pero sin nueva evidencia
- `evidence_based` — el cambio vino acompañado de claims con evidence_provided
  = 'source_cited' o 'document' en episodios intermedios

**Cómo implementarlo:**
Los datos ya están en DB. Para cada contradicción `direct` (data_artifact=0):

1. Identificar episodios intermedios entre episode_a y episode_b
2. Consultar claims intermedios con evidence_provided IN ('source_cited', 'document')
3. LLM evalúa si esa evidencia intermedia justifica el cambio de posición
4. Asigna change_type

**Prioridad:** Alta — es el diferenciador analítico central del dashboard
**Bloquea:** Fase E (visualización de contradicciones)
**Costo estimado:** ~$0.16 (16 contradicciones direct × ~$0.01 c/u)
**Caso ejemplo:** Ep237 ("Tyler actuó solo") ↔ Ep350 ("Tyler no cometió el asesinato")

## E.2 — Vista: Coherencia (scorecard narrativo)

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
