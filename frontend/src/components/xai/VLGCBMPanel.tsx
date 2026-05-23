import { useState } from "react";
import { Brain, Sparkles, ChevronDown, ChevronUp } from "lucide-react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
  Button,
  Badge,
} from "@/lib/shadcn-components";
import { useAnalysisStore } from "@/store/analysisStore";
import { chatApi } from "@/services/api";
import { cn } from "@/lib/utils";

function ConceptRow({
  label,
  value,
}: {
  label: string;
  value: number | string;
}) {
  const numVal = typeof value === "number" ? value : parseFloat(String(value));
  const isHigh = !isNaN(numVal) && numVal > 0.5;

  return (
    <div className="flex justify-between items-center text-xs py-1.5 border-b last:border-0">
      <span className="text-muted-foreground capitalize">
        {label.replace(/_/g, " ")}
      </span>
      <span
        className={cn(
          "font-mono font-medium",
          isHigh ? "text-orange-600 dark:text-orange-400" : "text-foreground",
        )}
      >
        {typeof value === "number" ? value.toFixed(4) : value}
      </span>
    </div>
  );
}

export function VLGCBMPanel() {
  const { sessionId, xaiResult, vlgCbmResult, setVlgCbmResult } =
    useAnalysisStore();
  const [isLoading, setIsLoading] = useState(false);
  const [isExpanded, setIsExpanded] = useState(true);

  if (!xaiResult) return null;

  const concepts = xaiResult.vlg_cbm_concepts;

  const generateExplanation = async () => {
    if (!sessionId) return;
    setIsLoading(true);
    try {
      const result = await chatApi.vlgCbm(sessionId);
      setVlgCbmResult(result);
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
          Conceptos XAI por región anatómica + explicación clínica con Gemini
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Conceptos */}
        {Object.keys(concepts).length > 0 && (
          <div className="rounded-lg border bg-muted/20 p-3">
            <button
              className="flex w-full items-center justify-between text-xs font-medium text-muted-foreground uppercase tracking-wide"
              onClick={() => setIsExpanded(!isExpanded)}
            >
              <span>Conceptos extraídos ({Object.keys(concepts).length})</span>
              {isExpanded ? (
                <ChevronUp className="h-3 w-3" />
              ) : (
                <ChevronDown className="h-3 w-3" />
              )}
            </button>
            {isExpanded && (
              <div className="mt-2 max-h-48 overflow-auto">
                {Object.entries(concepts).map(([k, v]) => (
                  <ConceptRow key={k} label={k} value={v as number | string} />
                ))}
              </div>
            )}
          </div>
        )}

        {/* Explicación Gemini */}
        {vlgCbmResult ? (
          <div className="rounded-lg border border-blue-200 dark:border-blue-800 bg-blue-50 dark:bg-blue-900/20 p-4 space-y-2">
            <div className="flex items-center gap-2">
              <Brain className="h-4 w-4 text-blue-600 dark:text-blue-400" />
              <span className="text-xs font-medium text-blue-700 dark:text-blue-300">
                Interpretación Clínica
              </span>
              <Badge variant="outline" className="text-[10px] ml-auto">
                {vlgCbmResult.model_used}
              </Badge>
            </div>
            <p className="text-sm text-blue-800 dark:text-blue-200 leading-relaxed">
              {vlgCbmResult.clinical_explanation}
            </p>
          </div>
        ) : (
          <Button
            variant="outline"
            className="w-full"
            onClick={generateExplanation}
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
                Generar Interpretación Clínica (Gemini)
              </>
            )}
          </Button>
        )}
      </CardContent>
    </Card>
  );
}
