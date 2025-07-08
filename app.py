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
    if "texto_hc" in datos:
        resumen_clinico = datos["texto_hc"]
    else:
        edad = str(datos.get("edad", "no informado"))
        antecedentes = ", ".join(datos.get("antecedentes", [])) or "sin antecedentes relevantes"
        factores = ", ".join(datos.get("riesgo", [])) or "sin factores informados"
        medicacion = ", ".join(datos.get("medicacion", [])) or "ninguna medicación registrada"
        laboratorio = ", ".join(f"{k}: {v}" for k, v in datos.get("laboratorio", {}).items()) or "sin estudios complementarios"

        resumen_clinico = (
            f"Paciente de {edad} años, con antecedentes de {antecedentes}, "
            f"factores de riesgo cardiovascular: {factores}, "
            f"actualmente medicado con {medicacion}, y presenta los siguientes estudios complementarios: {laboratorio}."
        )

    criterios_txt = "\n\n".join(
        f"{i+1}. {e['nombre']}: {e['descripcion']}\nCriterios: {e['criterios_texto']}"
        for i, e in enumerate(estudios)
    )

    prompt_final = f"""
Compara los datos recibidos del paciente con los criterios de inclusion de los estudios, y devolve como resultado solamente los estudios en que se cumplen todos los criterios o que solo falta uno.
Usá `✅` si cumple todos los criterios, `⚠️` si falta un criterio. Respuesta concreta y precisa en forma de lista por estudio.

---

📄 Datos del paciente:

{resumen_clinico}

---

📚 Estudios clínicos disponibles:
{criterios_txt}
""".strip()

    return prompt_final



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

