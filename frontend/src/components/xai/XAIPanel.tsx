import { useState, useEffect } from "react";
import { Layers, Play, Clock, Info, Eye, Cpu } from "lucide-react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
  Button,
  Checkbox,
  Label,
  Badge,
  Tooltip,
  TooltipContent,
  TooltipTrigger,
  TooltipProvider,
} from "@/lib/shadcn-components";
import { useAnalysisStore } from "@/store/analysisStore";
import {
  useXAI,
  usePrecomputedXAI,
  usePrecomputedStatus,
} from "@/hooks/useXAI";
import { XAI_METHOD_LABELS, XAI_METHOD_COLORS } from "@/types/api.types";
import { formatMs, cn } from "@/lib/utils";

const ALL_METHODS = [
  "gradcam",
  "integrated_gradients",
  "shap",
  "lime",
  "smoothgrad2",
];

const METHOD_HINTS: Record<string, string> = {
  gradcam:
    "Mapa de activación ponderado por gradiente. Rápido y representativo.",
  integrated_gradients:
    "Atribución por integral de camino. Más preciso que Grad-CAM.",
  shap: "Valores de Shapley. Base teórica sólida, más lento (~30-60 s).",
  lime: "Aproximación local por superpíxeles. Independiente del modelo, lento.",
  smoothgrad2:
    "Promedio del cuadrado de gradientes con ruido. Mapas más suaves y focalizados.",
};

type XAIMode = "precomputed" | "compute";

function XAIMapCard({ method, mapData }: { method: string; mapData: any }) {
  const label = XAI_METHOD_LABELS[method] || method;
  const colorClass = XAI_METHOD_COLORS[method] || "";

  if (mapData.timed_out) {
    return (
      <div className="flex flex-col rounded-lg border bg-muted/20 overflow-hidden">
        <div className="flex h-[220px] w-[220px] items-center justify-center bg-muted/30">
          <div className="text-center text-xs text-muted-foreground p-4">
            <Clock className="h-6 w-6 mx-auto mb-1 opacity-50" />
            Tiempo agotado
          </div>
        </div>
        <div className="p-2 text-center">
          <Badge variant="outline" className={cn("text-[10px]", colorClass)}>
            {label}
          </Badge>
        </div>
      </div>
    );
  }

  const src = mapData.heatmap_overlay_b64 || mapData.heatmap_b64;
  if (!src) return null;

  return (
    <div className="group flex flex-col rounded-lg border bg-card overflow-hidden hover:shadow-md transition-shadow">
      <div className="relative overflow-hidden bg-black">
        <img
          src={`data:image/png;base64,${src}`}
          alt={label}
          className="h-[220px] w-[220px] object-contain block mx-auto"
        />
      </div>
      <div className="p-2">
        <div className="text-center">
          <Badge className={cn("text-[10px] px-1.5 py-0", colorClass)}>
            {label}
          </Badge>
        </div>
      </div>
    </div>
  );
}

export function XAIPanel() {
  const {
    sessionId,
    stage,
    xaiResult,
    selectedXAIMethods,
    setSelectedXAIMethods,
  } = useAnalysisStore();
  const xai = useXAI();
  const precomputed = usePrecomputedXAI();
  const { data: status } = usePrecomputedStatus();

  const [mode, setMode] = useState<XAIMode>("precomputed");

  // Default to compute mode if no precomputed maps available
  useEffect(() => {
    if (status && !status.available) setMode("compute");
  }, [status]);

  const isComputing =
    stage === "xai_computing" || xai.isPending || precomputed.isPending;
  const canAct = !!sessionId && stage !== "idle" && !isComputing;

  const handleAction = () => {
    if (!sessionId) return;
    if (mode === "precomputed") {
      precomputed.mutate(sessionId);
    } else {
      xai.mutate({ sessionId, methods: selectedXAIMethods });
    }
  };

  const precomputedAvailable = status?.available ?? false;

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base flex items-center gap-2">
          <span className="flex h-6 w-6 items-center justify-center rounded bg-primary text-primary-foreground text-xs font-bold">
            4
          </span>
          Mapas de Explicabilidad (XAI)
        </CardTitle>
        <CardDescription>
          {mode === "precomputed"
            ? "Visualiza los mapas de atribución obtenidos del análisis del dataset de prueba en el notebook"
            : "Selecciona los métodos y calcula los mapas de atribución para la imagen cargada"}
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-5">
        {/* Pre-computed info */}
        {mode === "precomputed" && (
          <div className="rounded-md border bg-muted/30 px-3 py-2.5 text-xs text-muted-foreground space-y-1">
            {precomputedAvailable ? (
              <>
                <p className="font-medium text-foreground">
                  Mapas globales del dataset de prueba
                </p>
                <p>
                  Mapas de atribución promediados sobre el conjunto de prueba.
                  Se superponen sobre la imagen cargada junto con el contorno de
                  la predicción del modelo.
                </p>
                <br />
                <div className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                  Métodos disponibles:
                </div>
                <TooltipProvider>
                  <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                    {ALL_METHODS.map((method) => (
                      <Tooltip key={method}>
                        <TooltipTrigger asChild>
                          <div className="flex items-center gap-2 rounded-md border px-3 py-2 cursor-pointer hover:bg-muted/50 transition-colors">
                            <Label
                              htmlFor={method}
                              className="cursor-pointer text-sm flex-1"
                            >
                              {XAI_METHOD_LABELS[method]}
                            </Label>
                            <Info className="h-3 w-3 text-muted-foreground/50 shrink-0" />
                          </div>
                        </TooltipTrigger>
                        <TooltipContent side="right" className="max-w-[200px]">
                          <p className="text-xs">{METHOD_HINTS[method]}</p>
                        </TooltipContent>
                      </Tooltip>
                    ))}
                  </div>
                </TooltipProvider>
              </>
            ) : (
              <p>
                No hay mapas pre-calculados disponibles. Configura{" "}
                <code className="font-mono bg-muted px-1 rounded">
                  XAI_PRECOMPUTED_DIR
                </code>{" "}
                en el backend apuntando al directorio de checkpoints del
                notebook.
              </p>
            )}
          </div>
        )}

        {/* Action button */}
        <Button
          className="w-full"
          onClick={handleAction}
          disabled={
            !canAct ||
            (mode === "precomputed" && !precomputedAvailable) ||
            (mode === "compute" && selectedXAIMethods.length === 0)
          }
        >
          {isComputing ? (
            <>
              <span className="mr-2 inline-block h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
              {mode === "precomputed"
                ? "Cargando mapas XAI…"
                : `Calculando XAI (${selectedXAIMethods.length} métodos)…`}
            </>
          ) : mode === "precomputed" ? (
            <>
              <Eye className="mr-2 h-4 w-4" />
              Mostrar Mapas XAI
            </>
          ) : (
            <>
              <Play className="mr-2 h-4 w-4" />
              Calcular Mapas XAI
            </>
          )}
        </Button>

        {/* Results */}
        {xaiResult && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="text-xs font-medium text-muted-foreground uppercase tracking-wide flex items-center gap-1">
                <Layers className="h-3 w-3" /> Resultados
              </div>
              <Badge variant="secondary" className="text-xs">
                {formatMs(xaiResult.compute_time_ms)} ·{" "}
                {xaiResult.methods_available.length} mapas
              </Badge>
            </div>

            <div className="flex flex-wrap gap-3">
              {Object.entries(xaiResult.maps).map(([method, mapData]) => (
                <XAIMapCard key={method} method={method} mapData={mapData} />
              ))}
            </div>
          </div>
        )}

        {(xai.isError || precomputed.isError) && (
          <div className="rounded-md bg-destructive/10 border border-destructive/30 px-3 py-2 text-xs text-destructive">
            {mode === "precomputed"
              ? "No se pudieron cargar los mapas pre-calculados. Verifica que XAI_PRECOMPUTED_DIR esté configurado."
              : "Error al calcular XAI. Asegúrate de haber ejecutado la inferencia primero."}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
