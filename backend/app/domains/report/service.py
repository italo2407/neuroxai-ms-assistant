from datetime import datetime
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from app.core.session_store import SessionData
from app.shared.image_utils import (
    numpy_to_base64_png, overlay_mask_on_image,
    overlay_heatmap_on_image, overlay_heatmap_with_prediction,
    colorize_heatmap, resize_to,
)
from app.config import settings

TEMPLATES_DIR = Path(__file__).parent / "templates"


class ReportService:
    def __init__(self):
        self.env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)))

    def generate(self, session: SessionData, include_xai: bool,
                 include_vlg_cbm: bool, patient_label: str) -> bytes:
        """Render the HTML report and return UTF-8 bytes."""
        ctx = self._build_context(session, include_xai, include_vlg_cbm, patient_label)
        template = self.env.get_template("report.html.jinja2")
        return template.render(**ctx).encode("utf-8")

    def _build_context(self, session: SessionData, include_xai: bool,
                       include_vlg_cbm: bool, patient_label: str) -> dict:
        ctx = {
            "patient_label": patient_label,
            "session_id": session.session_id,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M UTC"),
            "model_info": "BasicUNetPlusPlus · 5-fold KFold",
            "metrics": session.metrics or {},
            "concepts": session.vlg_cbm_concepts or {},
            "include_xai_maps": include_xai,
            "include_vlg_cbm": include_vlg_cbm,
            "xai_maps": {},
            "clinical_explanation": "",
            "original_b64": None,
            "prediction_b64": None,
            "overlay_b64": None,
            "gt_b64": None,
        }

        # Images
        img_224 = None
        if session.image_np is not None:
            ctx["original_b64"] = numpy_to_base64_png(session.image_np)
            img_224 = resize_to(session.image_np, settings.image_size)
        if session.predicted_mask_np is not None:
            ctx["prediction_b64"] = numpy_to_base64_png(session.predicted_mask_np)
            if img_224 is not None:
                ctx["overlay_b64"] = overlay_mask_on_image(
                    img_224, session.predicted_mask_np, color=(255, 80, 80)
                )
        if session.gt_mask_np is not None:
            ctx["gt_b64"] = numpy_to_base64_png(session.gt_mask_np)

        # XAI maps — stored as raw float arrays in session
        if include_xai and session.xai_results:
            pred_224 = (
                resize_to(session.predicted_mask_np, settings.image_size)
                if session.predicted_mask_np is not None else None
            )

            class MapData:
                def __init__(self, heatmap_b64, timed_out=False):
                    self.heatmap_b64 = heatmap_b64
                    self.timed_out = timed_out

            for method, raw_map in session.xai_results.items():
                if raw_map is None:
                    ctx["xai_maps"][method] = MapData("", True)
                    continue
                if img_224 is not None and pred_224 is not None:
                    hb64 = overlay_heatmap_with_prediction(img_224, raw_map, pred_224, "hot")
                elif img_224 is not None:
                    hb64 = overlay_heatmap_on_image(img_224, raw_map, "hot")
                else:
                    hb64 = colorize_heatmap(raw_map, "hot")
                ctx["xai_maps"][method] = MapData(hb64)

        # VLG-CBM clinical explanation
        if include_vlg_cbm:
            from app.domains.chat.service import chat_service
            ctx["clinical_explanation"] = chat_service.generate_vlg_cbm_explanation(session)

        return ctx


report_service = ReportService()
