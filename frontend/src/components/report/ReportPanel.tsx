import { ExternalLink } from "lucide-react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
  Button,
  Label,
  Checkbox,
  Input,
} from "@/lib/shadcn-components";
import { useAnalysisStore } from "@/store/analysisStore";
import { useReport } from "@/hooks/useReport";
import { useState } from "react";

export function ReportPanel() {
  const { sessionId, stage } = useAnalysisStore();
  const { generate, isGenerating, error } = useReport();

  const [includeXai, setIncludeXai] = useState(true);
  const [includeClinicalInterpretation, setIncludeClinicalInterpretation] = useState(true);
  const [patientLabel, setPatientLabel] = useState("Paciente");

  const canGenerate = !!sessionId && stage !== "idle" && !isGenerating;

  const handleGenerate = () => {
    generate({
      include_xai_maps: includeXai,
      include_clinical_interpretation: includeClinicalInterpretation,
      patient_label: patientLabel,
    });
  };

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base flex items-center gap-2">
          <span className="flex h-6 w-6 items-center justify-center rounded bg-primary text-primary-foreground text-xs font-bold">
            6
          </span>
          Generar Informe
        </CardTitle>
        <CardDescription>
          Abre un informe clínico completo en una nueva pestaña
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Etiqueta paciente */}
        <div className="space-y-1.5">
          <Label
            htmlFor="patient-label"
            className="text-xs font-medium text-muted-foreground uppercase tracking-wide"
          >
            Etiqueta del Paciente
          </Label>
          <Input
            id="patient-label"
            value={patientLabel}
            onChange={(e) => setPatientLabel(e.target.value)}
            placeholder="p. ej. Paciente 081"
            className="h-9 text-sm"
          />
        </div>

        {/* Opciones */}
        <div className="space-y-2">
          <Label className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
            Incluir
          </Label>
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <Checkbox
                id="include-xai"
                checked={includeXai}
                onCheckedChange={(c) => setIncludeXai(c as boolean)}
              />
              <Label htmlFor="include-xai" className="text-sm cursor-pointer">
                Mapas XAI de atribución
              </Label>
            </div>
            <div className="flex items-center gap-2">
              <Checkbox
                id="include-clinical-interpretation"
                checked={includeClinicalInterpretation}
                onCheckedChange={(c) => setIncludeClinicalInterpretation(c as boolean)}
              />
              <Label htmlFor="include-clinical-interpretation" className="text-sm cursor-pointer">
                Interpretación clínica
              </Label>
            </div>
          </div>
        </div>

        <Button
          className="w-full"
          onClick={handleGenerate}
          disabled={!canGenerate}
        >
          {isGenerating ? (
            <>
              <span className="mr-2 inline-block h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
              Generando…
            </>
          ) : (
            <>
              <ExternalLink className="mr-2 h-4 w-4" />
              Ver Informe
            </>
          )}
        </Button>

        <p className="text-[10px] text-muted-foreground text-center">
          {sessionId
            ? "El informe se abrirá en una nueva pestaña · Usa Ctrl+P para guardar como PDF"
            : "Ejecuta la inferencia primero para generar el informe"}
        </p>

        {error && (
          <div className="rounded-md bg-destructive/10 border border-destructive/30 px-3 py-2 text-xs text-destructive">
            {error}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
