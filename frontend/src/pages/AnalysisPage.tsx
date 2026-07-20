import { UploadPanel } from "@/components/upload/UploadPanel";
import { ComparisonViewer } from "@/components/inference/ComparisonViewer";
import { MetricsPanel } from "@/components/inference/MetricsPanel";
import { XAIPanel } from "@/components/xai/XAIPanel";
import { ClinicalInterpretationPanel } from "@/components/xai/ClinicalInterpretationPanel";
import { ReportPanel } from "@/components/report/ReportPanel";
import { useAnalysisStore } from "@/store/analysisStore";
import { Brain, ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";

const STEPS = [
  { id: "upload", label: "Carga e Inferencia", stage: "idle" },
  { id: "compare", label: "Comparación" },
  { id: "metrics", label: "Métricas" },
  { id: "xai", label: "Mapas XAI" },
  { id: "clinical", label: "Interpretación Clínica" },
  { id: "report", label: "Informe" },
];

function StepBreadcrumb() {
  const { stage, inferenceResult, xaiResult } = useAnalysisStore();
  const active = inferenceResult ? (xaiResult ? 4 : 1) : 0;

  return (
    <nav className="flex items-center gap-1 text-xs text-muted-foreground overflow-x-auto pb-1 scrollbar-none">
      {STEPS.map((step, i) => (
        <div key={step.id} className="flex items-center gap-1 shrink-0">
          <span
            className={cn(
              "rounded-full px-2 py-0.5 transition-colors",
              i <= active
                ? "text-primary font-medium"
                : "text-muted-foreground",
            )}
          >
            {step.label}
          </span>
          {i < STEPS.length - 1 && <ChevronRight className="h-3 w-3" />}
        </div>
      ))}
    </nav>
  );
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center py-20 text-center gap-4">
      <div className="flex h-20 w-20 items-center justify-center rounded-2xl bg-muted">
        <Brain className="h-10 w-10 text-muted-foreground/50" />
      </div>
      <div>
        <h2 className="text-lg font-semibold">Inicia tu Análisis</h2>
        <p className="mt-1 text-sm text-muted-foreground max-w-sm">
          Carga una imagen MRI en escala de grises para ejecutar la segmentación
          de lesiones de EM con el ensemble 5-fold UNet++. Añade opcionalmente
          la máscara Ground Truth para calcular DICE e IoU.
        </p>
      </div>
      <div className="flex flex-wrap gap-2 justify-center mt-2">
        {[
          "Grad-CAM",
          "Gradientes Integrados",
          "SHAP",
          "LIME",
          "SmoothGrad²",
          "Chat Gemini",
        ].map((f) => (
          <span
            key={f}
            className="rounded-full border px-3 py-1 text-xs text-muted-foreground bg-muted/30"
          >
            {f}
          </span>
        ))}
      </div>
    </div>
  );
}

export function AnalysisPage() {
  const { inferenceResult } = useAnalysisStore();

  return (
    <div className="container mx-auto px-4 py-6 max-w-[1400px]">
      <div className="mb-4">
        <StepBreadcrumb />
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[340px_1fr]">
        {/* Sidebar */}
        <aside className="space-y-6">
          <UploadPanel />
          <ReportPanel />
        </aside>

        {/* Main content */}
        <main className="space-y-6">
          {!inferenceResult ? (
            <EmptyState />
          ) : (
            <>
              <ComparisonViewer />
              <MetricsPanel />
              <XAIPanel />
              <ClinicalInterpretationPanel />
            </>
          )}
        </main>
      </div>
    </div>
  );
}
