from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import json
import os
import httpx
import re
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# CORS para permitir conexión desde frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Podés restringir a ["https://medex.ar"] en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Endpoint principal de evaluación IA
@app.post("/evaluar_ia")
async def evaluar_ia(request: Request):
    try:
        data = await request.json()
        print("✅ JSON recibido:", data)

        texto_hc = data.get("datos", {}).get("texto_hc", "")

        with open("criterios_estudios_textual.json", "r", encoding="utf-8") as f:
            criterios = json.load(f)["estudios"]

        prompt = armar_prompt(texto_hc, criterios)
        print("📤 Enviando prompt estructurado a OpenRouter...")

        respuesta_ia = await consultar_openrouter(prompt)
        print("✅ Respuesta JSON recibida")

        json_match = re.search(r"```json\s*(\[.*?\])\s*```", respuesta_ia, re.DOTALL)
        if not json_match:
            raise ValueError("No se encontró JSON válido dentro del bloque ```json```.")

        json_parsed = json.loads(json_match.group(1))

        # Agregar descripciones desde archivo local
        descripciones = {e['nombre']: e['descripcion'] for e in criterios}
        for est in json_parsed:
            est['descripcion'] = descripciones.get(est['nombre'], "Sin descripción")

        return {"estudios": json_parsed}

    except Exception as e:
        print("❌ ERROR EN /evaluar_ia:", e)
        return {"error": str(e)}


# Endpoint adicional para exponer los criterios al frontend
@app.get("/criterios")
def obtener_criterios():
    try:
        with open("criterios_estudios_textual.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        return {"estudios": data["estudios"]}
    except Exception as e:
        print("❌ ERROR EN /criterios:", e)
        return {"error": str(e)}



def armar_prompt(texto_hc, estudios):
    def formatear_criterios(est):
        criterios = est["criterios"]
        texto = "Criterios de inclusión:\n"
        for c in criterios.get("inclusion", []):
            if c["tipo"] == "grupo-condicional":
                texto += f"- {c['condicion']}:\n"
                for opc in c["opciones"]:
                    texto += f"    • {opc['condicion']}\n"
            else:
                texto += f"- {c['condicion']}\n"

        if criterios.get("exclusion"):
            texto += "\nCriterios de exclusión:\n"
            for c in criterios["exclusion"]:
                texto += f"- {c['condicion']}\n"
        return texto.strip()

    criterios_txt = "\n\n".join(
        f"{i+1}. {e['nombre']}: {e['descripcion']}\n{formatear_criterios(e)}"
        for i, e in enumerate(estudios)
    )

    prompt_final = f"""
Sos un asistente clínico experto. Evaluá los siguientes estudios clínicos y decidí si un paciente califica.

### Instrucciones:
- Analizá los criterios de inclusión (todos deben cumplirse) y exclusión (si uno se cumple, queda excluido).
- Los criterios pueden incluir condiciones numéricas, diagnósticos, medicación o grupos condicionales (por ejemplo, al menos 2 de una lista).
- Respondé solamente si el paciente cumple totalmente (✅) o parcialmente (⚠️). No incluyas estudios no aplicables (❌).
- Si un dato no está mencionado en el texto clínico, asumí que NO está presente.
- Devolvé un array JSON con los siguientes campos por estudio:

```json
[
  {{
    "nombre": "Nombre del estudio",
    "descripcion": "",
    "estado": "✅ / ⚠️",    
    "detalle": "Motivo del cumplimiento parcial o total"
  }}
]
---

Texto clínico del paciente:
{texto_hc}

---

Estudios clínicos disponibles:
{criterios_txt}
""".strip()

    return prompt_final


async def consultar_openrouter(prompt):
    try:
        OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
        if not OPENROUTER_API_KEY:
            raise ValueError("Falta la API KEY de OpenRouter en variables de entorno.")

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "https://medex.ar",
            "Content-Type": "application/json"
        }

        body = {
            "model": "cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
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


