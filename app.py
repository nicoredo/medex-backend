from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
import os
import httpx

app = FastAPI()

# CORS para frontend local
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ruta principal
@app.post("/evaluar_ia")
async def evaluar_ia(request: Request):
    try:
        data = await request.json()
        print("✅ JSON recibido:", data)

        datos_paciente = data.get("datos", {})

        with open("criterios_estudios_textual.json", "r", encoding="utf-8") as f:
            criterios = json.load(f)["estudios"]

        prompt = armar_prompt(datos_paciente, criterios)
        print("📤 Enviando prompt a OpenRouter...")

        respuesta_ia = await consultar_openrouter(prompt)
        print("✅ Respuesta de OpenRouter recibida")

        return {"resumen_html": respuesta_ia}

    except Exception as e:
        print("❌ ERROR EN /evaluar_ia:", e)
        return {"error": str(e)}


def armar_prompt(datos, estudios):
    def format_list(label, items):
        return f"- {label}: {', '.join(items) if items else 'No informado'}"

    def format_lab(lab):
        if not lab:
            return "- Estudios complementarios: No informado"
        return "- Estudios complementarios:\n  " + "\n  ".join(f"{k}: {v}" for k, v in lab.items())

    resumen = f"""
📄 Datos del paciente:

{format_list("Edad", [str(datos.get("edad"))] if datos.get("edad") else [])}
{format_list("Antecedentes", datos.get("antecedentes", []))}
{format_list("Factores de riesgo", datos.get("riesgo", []))}
{format_list("Medicación", datos.get("medicacion", []))}
{format_lab(datos.get("laboratorio", {}))}
""".strip()

    criterios_txt = "\n\n".join(
        f"{i+1}. {e['nombre']}: {e['descripcion']}\nCriterios: {e['criterios_texto']}"
        for i, e in enumerate(estudios)
    )

    prompt_final = f"""
Sos un asistente clínico experto.
Tu tarea es evaluar si un paciente califica para un estudio médico específico basado en sus datos clínicos (edad, antecedentes, factores de riesgo, medicacion y estudios complementarios).
- Responde con un resumen claro, solo menciona los estudios que no están excluidos, y que tienen al menos dos criterios cumplidos.
- Formato de salida: HTML simple con "Cumple" o "Cumple parcial" íconos (✅, ⚠️), título del estudio y breve observación de datos faltantes en los parciales.
- No incluyas información adicional, solo el resumen de los estudios necesarios.

---

{resumen}

---

📚 Estudios disponibles:
{criterios_txt}
"""

    return prompt_final.strip()


async def consultar_openrouter(prompt):
    try:

        OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

headers = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "HTTP-Referer": "http://localhost",
    "Content-Type": "application/json"
}


        body = {
            "model": "openai/gpt-4o",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 1000
        }

        async with httpx.AsyncClient() as client:
            r = await client.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=body)
            print("📨 STATUS:", r.status_code)
            print("📨 TEXT:", r.text)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]

    except Exception as e:
        print("❌ ERROR EN consultar_openrouter:", e)
        raise


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)

