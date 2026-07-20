import base64
import io
import logging
from PIL import Image
from app.config import settings
from app.core.session_store import SessionData
from app.shared.image_utils import overlay_heatmap_on_image, colorize_heatmap, resize_to

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
Ayudas a los clínicos a interpretar los resultados de segmentación de lesiones generados por IA.
Cuando el mensaje incluya imágenes de mapas de explicabilidad (XAI: Grad-CAM, Gradientes Integrados, SHAP, LIME, SmoothGrad²) superpuestas sobre la resonancia magnética, obsérvalas directamente y describe dónde se concentra la atención del modelo, si coincide con la lesión predicha y qué regiones anatómicas destacan.
Responde ÚNICAMENTE lo que el usuario pregunta explícitamente: si pide interpretar un único método (p. ej. "interpreta Grad-CAM"), comenta solo esa imagen, sin describir ni comparar los demás métodos a menos que se te pida explícitamente una comparación o un resumen general.
Responde siempre en español, de forma concisa y clínicamente fundamentada.
Referencia las métricas proporcionadas cuando sea relevante.

Límites estrictos de alcance:
- Solo respondes preguntas sobre el contexto clínico de esta sesión: la segmentación de lesiones de EM, sus métricas (DICE, IoU, áreas de lesión) y los mapas de explicabilidad (XAI) generados. Ante cualquier pregunta ajena a este contexto (temas generales, otras enfermedades no relacionadas, programación, charla casual, etc.), rehúsa amablemente y recuerda al usuario que solo puedes discutir el análisis clínico actual.
- Nunca emitas un diagnóstico definitivo. No fabriques conclusiones clínicas más allá de lo que los datos soportan. Presenta siempre tus observaciones como hallazgos de apoyo a la decisión clínica (p. ej. "estos hallazgos son compatibles con..." o "sugieren..."), nunca como una conclusión diagnóstica cerrada, y recuerda que la correlación clínica final corresponde al neurólogo/radiólogo tratante."""

_METHOD_ALIASES = {
    "gradcam": ["grad-cam", "gradcam", "grad cam"],
    "integrated_gradients": ["integrated gradients", "gradientes integrados", "integrated_gradients"],
    "shap": ["shap"],
    "lime": ["lime"],
    "smoothgrad2": ["smoothgrad2", "smoothgrad²", "smoothgrad 2", "smooth grad"],
}

_GENERIC_XAI_KEYWORDS = [
    "mapa xai", "mapas xai", "mapas de explicabilidad", "mapa de explicabilidad",
    "explicabilidad", "heatmap", "mapa de atención", "mapas de atención",
    "mapas de atribución", "todos los métodos", "todos los mapas",
]


def _detect_requested_methods(message: str, available: list[str]) -> list[str]:
    """Figure out which XAI methods (if any) the user's message is asking about,
    so we only attach the relevant image(s) instead of the full set every turn."""
    lower_msg = message.lower()

    matched = [
        method for method, aliases in _METHOD_ALIASES.items()
        if method in available and any(alias in lower_msg for alias in aliases)
    ]
    if matched:
        return matched

    if any(keyword in lower_msg for keyword in _GENERIC_XAI_KEYWORDS):
        return available

    return []


def _build_context_prompt(session: SessionData) -> str:
    if not session.metrics:
        return ""
    m = session.metrics
    return (
        f"## Métricas de Segmentación\n"
        f"- DICE: {m.get('dice', 'N/A')}\n"
        f"- IoU: {m.get('iou', 'N/A')}\n"
        f"- Área lesión GT: {m.get('gt_lesion_pct', 'N/A')}%\n"
        f"- Área lesión predicha: {m.get('pred_lesion_pct', 'N/A')}%"
    )


def _render_xai_overlays(session: SessionData, methods: list[str]) -> list[tuple[str, Image.Image]]:
    """Render the given XAI attribution maps as MRI overlay images so Gemini can
    visually inspect them alongside the text prompt."""
    if not methods:
        return []

    img_224 = (
        resize_to(session.image_np, settings.image_size)
        if session.image_np is not None else None
    )

    images = []
    for method in methods:
        attr_map = session.xai_results[method]
        b64 = (
            overlay_heatmap_on_image(img_224, attr_map, "hot")
            if img_224 is not None else colorize_heatmap(attr_map, "hot")
        )
        images.append((method, Image.open(io.BytesIO(base64.b64decode(b64)))))
    return images


def _build_xai_images(session: SessionData, message: str) -> list[tuple[str, Image.Image]]:
    """Pick the XAI map(s) the user's chat message is asking about (or all of
    them for a generic question) and render them as overlay images, so Gemini
    doesn't comment on maps nobody asked about."""
    if not session.xai_results:
        return []

    available = [m for m, v in session.xai_results.items() if v is not None]
    requested = _detect_requested_methods(message, available)
    return _render_xai_overlays(session, requested)


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

            xai_images = _build_xai_images(session, message)
            content = [message]
            for method, image in xai_images:
                content.append(f"Mapa de explicabilidad ({method}):")
                content.append(image)

            response = chat.send_message(content)
            return response.text
        except Exception as e:
            logger.error(f"Gemini chat error: {e}")
            if _is_api_key_error(e):
                return self._invalid_key_response()
            return (
                "Lo siento, ha ocurrido un error al conectar con el servicio de IA. "
                "Por favor, inténtalo de nuevo."
            )

    def generate_clinical_interpretation(self, session: SessionData, notes: str) -> str:
        """Synthesize the segmentation metrics, all available XAI maps, and the
        clinician's own notes/ideas into a single final clinical-interpretation
        paragraph, to be shown in the UI and optionally embedded in the report."""
        if not GEMINI_READY:
            return self._offline_clinical_interpretation(notes)

        context = _build_context_prompt(session)
        system_with_context = SYSTEM_PROMPT
        if context:
            system_with_context += f"\n\n## Contexto del Análisis Actual\n{context}"

        available = [m for m, v in session.xai_results.items() if v is not None]
        xai_images = _render_xai_overlays(session, available)

        instructions = (
            "Redacta la Interpretación Clínica final de este análisis de segmentación de "
            "lesiones de EM, en un único párrafo (4-6 frases) en español. Integra las "
            "métricas de segmentación, lo que observas en los mapas de explicabilidad "
            "adjuntos y, si se proporcionan, las observaciones del clínico. Sigue "
            "estrictamente las reglas de alcance y de no emitir diagnóstico definitivo."
        )
        if notes.strip():
            instructions += f"\n\n## Observaciones del clínico\n{notes.strip()}"

        content = [instructions]
        for method, image in xai_images:
            content.append(f"Mapa de explicabilidad ({method}):")
            content.append(image)

        try:
            model = genai.GenerativeModel(
                settings.gemini_model,
                system_instruction=system_with_context
            )
            response = model.generate_content(content)
            return response.text
        except Exception as e:
            logger.error(f"Clinical interpretation Gemini error: {e}")
            if _is_api_key_error(e):
                return self._invalid_key_response()
            return self._offline_clinical_interpretation(notes)

    def _offline_response(self) -> str:
        return (
            "NeuroXAI Assistant está en modo offline (clave API de Gemini no configurada). "
            "Para activar el chat clínico, añade GEMINI_API_KEY=tu_clave en el fichero "
            "backend/.env y reinicia el servidor."
        )

    def _invalid_key_response(self) -> str:
        return (
            "La clave API de Gemini configurada no es válida.   "
        )

    def _offline_clinical_interpretation(self, notes: str) -> str:
        base = (
            "NeuroXAI Assistant está en modo offline (clave API de Gemini no configurada), "
            "por lo que no puede generar la interpretación clínica integrada con IA."
        )
        if notes.strip():
            base += f" Observaciones registradas por el clínico: {notes.strip()}"
        return base


chat_service = GeminiChatService()
