
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
import httpx

app = FastAPI()

# Permitir cualquier origen (modo desarrollo)
app.add_middleware(
    CORSMiddleware,
allow_origin_regex=r"http://medex\\.ar",    
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
def home():
    return {"status": "ok"}

MISTRAL_API_KEY = "HmK9kErc0JZUwZcdeksBJujc7lF5U4iz"
MISTRAL_ENDPOINT = "https://api.mistral.ai/v1/chat/completions"


@app.post("/evaluar_ia")
async def evaluar_ia(request: Request):
    data = await request.json()
    print("📩 Recibido en /evaluar_ia:", data)
    datos_paciente = data.get("datos", {})

    with open("criterios_estudios_textual.json", "r", encoding="utf-8") as f:
        criterios = json.load(f)["estudios"]

    prompt = armar_prompt(datos_paciente, criterios)
    respuesta_ia = await consultar_mistral(prompt)
    return {"resumen_html": respuesta_ia}

@app.post("/chat_ia")
async def chat_ia(request: Request):
    data = await request.json()
    print("📩 Recibido en /chat_ia:", data)
    pregunta = data.get("pregunta")
    datos = data.get("datos")

    if not pregunta or not datos:
        return {"respuesta": "Faltan datos o pregunta."}

    resumen = construir_resumen_paciente(datos)

    prompt = f"""
Sos un asistente clínico. El siguiente paciente fue analizado con IA, y ahora el médico desea refinar la evaluación.

{resumen}

❓ Pregunta del médico: {pregunta}
Por favor, respondé de forma clara y útil para el profesional.
"""

    respuesta = await consultar_mistral(prompt)
    return {"respuesta": respuesta}

def construir_resumen_paciente(datos):
    edad = datos.get("edad")
    try:
        edad = int(edad)
    except:
        edad = "-"

    antecedentes = ", ".join(datos.get("antecedentes", [])) or "-"
    riesgo = ", ".join(datos.get("riesgo", [])) or "-"
    medicacion = ", ".join(datos.get("medicacion", [])) or "-"
    lab = ", ".join(f"{k}: {v}" for k, v in datos.get("laboratorio", {}).items()) or "-"

    return f"""
📄 Datos del paciente:
Edad: {edad}
Antecedentes: {antecedentes}
Factores de riesgo: {riesgo}
Medicación: {medicacion}
Estudios complementarios: {lab}
"""

def armar_prompt(datos, estudios):
    resumen = construir_resumen_paciente(datos)
    criterios_txt = "\n\n".join(
        f"{i+1}. {e['nombre']}: {e['descripcion']}\nCriterios: {e['criterios_texto']}"
        for i, e in enumerate(estudios)
    )

    prompt_final = f"""
Sos un evaluador clínico inteligente. Tu tarea es analizar si el paciente cumple los criterios de inclusión y exclusión de cada estudio.

Clasificá cada estudio como:
✅ Cumple
⚠️ Parcial (falta algún criterio leve)
❌ Excluido (no cumple criterios importantes o algún criterio excluyente)

Solo devolver los estudios cumplidos total o parcialmente, en HTML para mostrar en una web.

---

{resumen}

---

📚 Estudios:
{criterios_txt}
"""
    return prompt_final.strip()

async def consultar_mistral(prompt):
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }

    body = {
        "model": "mistral-medium",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 700
    }

    async with httpx.AsyncClient() as client:
        r = await client.post(MISTRAL_ENDPOINT, headers=headers, json=body)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
