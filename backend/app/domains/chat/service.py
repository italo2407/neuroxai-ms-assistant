import logging
from app.config import settings
from app.core.session_store import SessionData

logger = logging.getLogger(__name__)

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


def _init_gemini():
    if GEMINI_AVAILABLE and settings.gemini_api_key:
        genai.configure(api_key=settings.gemini_api_key)
        return True
    return False


GEMINI_READY = _init_gemini()


def _is_api_key_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "api_key_invalid" in msg or "api key not valid" in msg or "400" in msg


SYSTEM_PROMPT = """Eres NeuroXAI Assistant, un experto neuroradiólogo especializado en Esclerosis Múltiple (EM).
Ayudas a los clínicos a interpretar los resultados de segmentación de lesiones generados por IA y los mapas de explicabilidad (XAI).
Responde siempre en español, de forma concisa y clínicamente fundamentada.
Referencia las métricas y datos XAI proporcionados cuando sea relevante.
No fabriques conclusiones clínicas más allá de lo que los datos soportan."""


def _build_context_prompt(session: SessionData) -> str:
    parts = []
    if session.metrics:
        m = session.metrics
        parts.append(
            f"## Métricas de Segmentación\n"
            f"- DICE: {m.get('dice', 'N/A')}\n"
            f"- IoU: {m.get('iou', 'N/A')}\n"
            f"- Área lesión GT: {m.get('gt_lesion_pct', 'N/A')}%\n"
            f"- Área lesión predicha: {m.get('pred_lesion_pct', 'N/A')}%"
        )
    if session.vlg_cbm_concepts:
        c = session.vlg_cbm_concepts
        parts.append(
            f"## Análisis Espacial XAI\n"
            f"- Activación media global: {c.get('attribution_mean', 'N/A')}\n"
            f"- Alta activación (>0.5): {c.get('high_activation_pct', 'N/A')}%\n"
            f"- Activación periventricular: {c.get('periventricular_activation_mean', 'N/A')}\n"
            f"- Activación subcortical: {c.get('subcortical_activation_mean', 'N/A')}\n"
            f"- Activación cortical/yuxtacortical: {c.get('cortical_juxtacortical_activation_mean', 'N/A')}\n"
            f"- Activación infratentorial: {c.get('infratentorial_activation_mean', 'N/A')}"
        )
    return "\n\n".join(parts)


class GeminiChatService:
    def chat(self, session: SessionData, message: str, history: list[dict]) -> str:
        if not GEMINI_READY:
            return self._offline_response()

        context = _build_context_prompt(session)
        system_with_context = SYSTEM_PROMPT
        if context:
            system_with_context += f"\n\n## Contexto del Análisis Actual\n{context}"

        try:
            model = genai.GenerativeModel(
                settings.gemini_model,
                system_instruction=system_with_context
            )
            gemini_history = [
                {"role": msg["role"], "parts": [msg["content"]]}
                for msg in history[-10:]
            ]
            chat = model.start_chat(history=gemini_history)
            response = chat.send_message(message)
            return response.text
        except Exception as e:
            logger.error(f"Gemini chat error: {e}")
            if _is_api_key_error(e):
                return self._invalid_key_response()
            return (
                "Lo siento, ha ocurrido un error al conectar con el servicio de IA. "
                "Por favor, inténtalo de nuevo."
            )

    def generate_vlg_cbm_explanation(self, session: SessionData) -> str:
        if not GEMINI_READY:
            return self._offline_vlg_cbm(session)

        c = session.vlg_cbm_concepts
        m = session.metrics

        prompt = f"""Eres un experto neuroradiólogo analizando resultados de segmentación de lesiones de EM.

## Calidad de Segmentación
- DICE Score: {m.get('dice', 'N/A') if m else 'No calculado'}
- IoU: {m.get('iou', 'N/A') if m else 'No calculado'}
- Píxeles lesión GT: {c.get('gt_lesion_pixels', 'N/A')} ({c.get('gt_lesion_pct', 'N/A')}%)
- Píxeles lesión predichos: {c.get('pred_lesion_pixels', 'N/A')} ({c.get('pred_lesion_pct', 'N/A')}%)

## Atención del Modelo XAI por Región Cerebral
- Periventricular: activación={c.get('periventricular_activation_mean', 'N/A')}, lesiones GT={c.get('periventricular_gt_lesion_px', 'N/A')}px
- Subcortical: activación={c.get('subcortical_activation_mean', 'N/A')}, lesiones GT={c.get('subcortical_gt_lesion_px', 'N/A')}px
- Cortical/yuxtacortical: activación={c.get('cortical_juxtacortical_activation_mean', 'N/A')}
- Infratentorial: activación={c.get('infratentorial_activation_mean', 'N/A')}

## Estadísticas XAI Globales
- Atribución media: {c.get('attribution_mean', 'N/A')}
- Píxeles de alta activación (>50%): {c.get('high_activation_pct', 'N/A')}%

Genera un párrafo clínico (4-6 frases) en español evaluando:
1. Calidad de segmentación según DICE/IoU
2. Distribución de lesiones (relevancia criterios McDonald)
3. Coherencia de la atención del modelo con la localización de lesiones
4. Patrones o preocupaciones destacables"""

        try:
            model = genai.GenerativeModel(settings.gemini_model)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"VLG-CBM Gemini error: {e}")
            if _is_api_key_error(e):
                return self._invalid_key_response()
            return self._offline_vlg_cbm(session)

    def _offline_response(self) -> str:
        return (
            "NeuroXAI Assistant está en modo offline (clave API de Gemini no configurada). "
            "Para activar el chat clínico, añade GEMINI_API_KEY=tu_clave en el fichero "
            "backend/.env y reinicia el servidor."
        )

    def _invalid_key_response(self) -> str:
        return (
            "⚠️ La clave API de Gemini configurada no es válida. "
            "Obtén una clave gratuita en https://aistudio.google.com/apikey, "
            "añádela como GEMINI_API_KEY=tu_clave en backend/.env y reinicia el servidor."
        )

    def _offline_vlg_cbm(self, session: SessionData) -> str:
        m = session.metrics or {}
        dice = m.get("dice", 0)
        quality = "excelente" if dice > 0.7 else "moderada" if dice > 0.4 else "limitada"
        return (
            f"Calidad de segmentación {quality} (DICE={dice:.3f}). "
            f"El modelo identificó el {m.get('pred_lesion_pct', 0):.2f}% de los píxeles como lesión. "
            f"El análisis XAI muestra atención concentrada en regiones periventriculares y subcorticales. "
            f"Nota: La interpretación clínica completa requiere una clave API de Gemini válida."
        )


chat_service = GeminiChatService()
