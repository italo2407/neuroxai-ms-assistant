import { useState } from "react";
import { Brain, Sparkles } from "lucide-react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
  Button,
  Badge,
  Label,
  Textarea,
} from "@/lib/shadcn-components";
import { useAnalysisStore } from "@/store/analysisStore";
import { chatApi } from "@/services/api";

export function ClinicalInterpretationPanel() {
  const {
    sessionId, xaiResult, clinicalInterpretation, setClinicalInterpretation,
  } = useAnalysisStore();
  const [notes, setNotes] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  if (!xaiResult) return null;

  const generateInterpretation = async () => {
    if (!sessionId) return;
    setIsLoading(true);
    try {
      const result = await chatApi.generateClinicalInterpretation(sessionId, notes);
      setClinicalInterpretation(result);
    } catch (e) {
      console.error(e);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base flex items-center gap-2">
          <span className="flex h-6 w-6 items-center justify-center rounded bg-primary text-primary-foreground text-xs font-bold">
            5
          </span>
          Interpretación Clínica
        </CardTitle>
        <CardDescription>
          Añade tus observaciones y genera la interpretación clínica final con Gemini
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-1.5">
          <Label className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
            Tus observaciones (opcional)
          </Label>
          <Textarea
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            placeholder="p. ej. Lesión periventricular de nueva aparición, sospecha de brote…"
            className="min-h-[80px] text-sm resize-none"
          />
        </div>

        {clinicalInterpretation && (
          <div className="rounded-lg border border-blue-200 dark:border-blue-800 bg-blue-50 dark:bg-blue-900/20 p-4 space-y-2">
            <div className="flex items-center gap-2">
              <Brain className="h-4 w-4 text-blue-600 dark:text-blue-400" />
              <span className="text-xs font-medium text-blue-700 dark:text-blue-300">
                Interpretación Clínica
              </span>
              <Badge variant="outline" className="text-[10px] ml-auto">
                {clinicalInterpretation.model_used}
              </Badge>
            </div>
            <p className="text-sm text-blue-800 dark:text-blue-200 leading-relaxed whitespace-pre-line">
              {clinicalInterpretation.interpretation}
            </p>
          </div>
        )}

        <Button
          variant="outline"
          className="w-full"
          onClick={generateInterpretation}
          disabled={isLoading}
        >
          {isLoading ? (
            <>
              <span className="mr-2 inline-block h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
              Generando interpretación clínica…
            </>
          ) : (
            <>
              <Sparkles className="mr-2 h-4 w-4" />
              {clinicalInterpretation
                ? "Regenerar Interpretación Clínica (Gemini)"
                : "Generar Interpretación Clínica (Gemini)"}
            </>
          )}
        </Button>
      </CardContent>
    </Card>
  );
}
