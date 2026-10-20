# Diagrama E6 — Chatbot de reglas

```mermaid
flowchart TD
    A[Saludo, privacidad y límites] --> B[Preguntar siguiente slot]
    B --> C{Respuesta válida?}
    C -- No --> D{Tres intentos agotados?}
    D -- No --> E[Explicar error, dar ejemplo y repetir]
    E --> B
    D -- Sí --> F[Derivar a tutoría y borrar datos]
    C -- Sí --> G[Guardar respuesta temporal]
    G --> H{Ocho variables completas?}
    H -- No --> B
    H -- Sí --> I[POST /predecir]
    I --> J{Respuesta válida?}
    J -- No --> K[Informar fallo, sugerir tutoría y borrar datos]
    J -- Sí --> L{Dropout o P Dropout supera umbral validado?}
    L -- Sí --> M[Explicar estimación y derivar a tutoría]
    L -- No --> N[Dar orientación no concluyente]
    M --> O[Borrar datos y terminar]
    N --> O
```

La comparación probabilística solo se realiza si E4 proporciona un umbral validado; sin umbral, se informa la limitación y se aconseja revisión por tutoría. No se almacenan nombres, DNI ni respuestas tras finalizar.
