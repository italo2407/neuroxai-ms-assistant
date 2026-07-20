from pydantic import BaseModel
from typing import Optional


class ReportRequest(BaseModel):
    session_id: str
    include_xai_maps: bool = True
    include_clinical_interpretation: bool = True
    patient_label: Optional[str] = "Paciente"
