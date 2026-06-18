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
