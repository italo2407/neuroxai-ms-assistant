import { useState } from "react";
import { ZoomIn, ZoomOut, RotateCcw } from "lucide-react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
  Tabs,
  TabsList,
  TabsTrigger,
  TabsContent,
  Badge,
  Button,
} from "@/lib/shadcn-components";
import { useAnalysisStore } from "@/store/analysisStore";
import { getDiceColor, getDiceLabel, cn } from "@/lib/utils";

/* ─── Zoom hook ─── */
function useZoom(initial = 1) {
  const [scale, setScale] = useState(initial);
  const zoomIn = () => setScale((s) => Math.min(s + 0.25, 3));
  const zoomOut = () => setScale((s) => Math.max(s - 0.25, 0.5));
  const reset = () => setScale(1);
  return { scale, zoomIn, zoomOut, reset };
}

/* ─── Imagen individual con zoom ─── */
function ZoomableImage({
  src,
  alt,
  label,
  badge,
  subtitle,
}: {
  src: string;
  alt: string;
  label: string;
  badge?: string;
  subtitle?: string;
}) {
  const { scale, zoomIn, zoomOut, reset } = useZoom();

  return (
    <div className="flex flex-col items-center gap-2">
      <div className="flex items-center gap-1">
        <Button
          variant="ghost"
          size="icon"
          className="h-6 w-6"
          onClick={zoomOut}
          title="Alejar"
        >
          <ZoomOut className="h-3 w-3" />
        </Button>
        <span className="text-[10px] text-muted-foreground w-8 text-center">
          {Math.round(scale * 100)}%
        </span>
        <Button
          variant="ghost"
          size="icon"
          className="h-6 w-6"
          onClick={zoomIn}
          title="Acercar"
        >
          <ZoomIn className="h-3 w-3" />
        </Button>
      </div>

      <div
        className="overflow-hidden rounded-lg border bg-black/5 dark:bg-white/5 flex justify-center items-center"
        style={{ width: 240, height: 240 }}
      >
        <img
          src={src}
          alt={alt}
          className="mri-image transition-transform duration-200"
          style={{
            width: 200,
            height: 240,
            transform: `scale(${scale})`,
            transformOrigin: "center center",
          }}
        />
      </div>

      <div className="flex flex-col items-center gap-0.5">
        <span className="text-xs font-medium text-muted-foreground">
          {label}
        </span>
        {badge && (
          <Badge variant="secondary" className="text-[10px] px-1.5 py-0">
            {badge}
          </Badge>
        )}
        {subtitle && (
          <span className="text-[10px] text-muted-foreground">{subtitle}</span>
        )}
      </div>
    </div>
  );
}

/* ─── Tarjeta de métrica ─── */
function MetricChip({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color: string;
}) {
  return (
    <div className="rounded-lg border bg-muted/30 px-4 py-2 text-center min-w-[90px]">
      <div className="text-[11px] text-muted-foreground">{label}</div>
      <div className={cn("text-lg font-bold", color)}>{value}</div>
    </div>
  );
}

/* ─── Leyenda FP/FN ─── */
function FPFNLegend() {
  return (
    <div className="flex flex-wrap gap-3 justify-center text-xs">
      {[
        { color: "bg-green-500", label: "VP (Verdadero Positivo)" },
        { color: "bg-red-500", label: "FP (Falso Positivo)" },
        { color: "bg-blue-500", label: "FN (Falso Negativo)" },
        { color: "bg-gray-400", label: "VN (fondo)" },
      ].map(({ color, label }) => (
        <div key={label} className="flex items-center gap-1.5">
          <span className={cn("inline-block h-3 w-3 rounded-sm", color)} />
          <span className="text-muted-foreground">{label}</span>
        </div>
      ))}
    </div>
  );
}

/* ─── Componente principal ─── */
export function ComparisonViewer() {
  const { inferenceResult, imagePreviewUrl, metrics } = useAnalysisStore();

  if (!inferenceResult) return null;

  const hasFPFN = !!inferenceResult.fpfn_overlay_b64;
  const hasGT = inferenceResult.has_gt;

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base flex items-center gap-2">
          <span className="flex h-6 w-6 items-center justify-center rounded bg-primary text-primary-foreground text-xs font-bold">
            2
          </span>
          Predicción vs Ground Truth
        </CardTitle>
        <CardDescription>
          Comparación lado a lado de la MRI, máscara predicha y GT (si está
          disponible)
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <Tabs defaultValue="comparacion">
          <TabsList className="mb-4 flex-wrap h-auto gap-1">
            <TabsTrigger value="comparacion">Comparación</TabsTrigger>
            {hasFPFN && <TabsTrigger value="fpfn">FP / FN</TabsTrigger>}
            <TabsTrigger value="folds">
              K-Fold ({inferenceResult.fold_masks_b64.length})
            </TabsTrigger>
          </TabsList>

          {/* ── Tab: Comparación (fusiona antigua Comparación + Superposición) ── */}
          <TabsContent value="comparacion">
            <div className="space-y-6">
              <div className="flex flex-wrap gap-6 justify-center">
                {imagePreviewUrl && (
                  <ZoomableImage
                    src={imagePreviewUrl}
                    alt="MRI original"
                    label="MRI Original"
                  />
                )}
                <ZoomableImage
                  src={`data:image/png;base64,${inferenceResult.overlay_b64}`}
                  alt="Predicción superpuesta"
                  label="Predicción sobre MRI"
                  badge="Rojo = Lesión predicha"
                />
                {hasGT && inferenceResult.gt_overlay_b64 && (
                  <ZoomableImage
                    src={`data:image/png;base64,${inferenceResult.gt_overlay_b64}`}
                    alt="Ground Truth superpuesta"
                    label="GT sobre MRI"
                    badge="Azul = Lesión GT"
                  />
                )}
              </div>

              {/* {metrics && (
                <div className="space-y-3">
                  <div className="text-xs font-medium text-muted-foreground uppercase tracking-wide text-center">
                    Métricas de evaluación
                  </div>
                  <div className="flex flex-wrap gap-2 justify-center">
                    <MetricChip
                      label="DICE"
                      value={metrics.dice.toFixed(4)}
                      color={getDiceColor(metrics.dice)}
                    />
                    <MetricChip
                      label="IoU"
                      value={metrics.iou.toFixed(4)}
                      color="text-blue-600 dark:text-blue-400"
                    />
                    <MetricChip
                      label="Precisión"
                      value={metrics.precision.toFixed(4)}
                      color="text-purple-600 dark:text-purple-400"
                    />
                    <MetricChip
                      label="Recall"
                      value={metrics.recall.toFixed(4)}
                      color="text-orange-600 dark:text-orange-400"
                    />
                  </div>
                  <div className={cn(
                    "flex items-center justify-center gap-2 rounded-lg p-2.5 text-sm font-medium",
                    metrics.dice >= 0.7
                      ? "bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-300"
                      : metrics.dice >= 0.4
                      ? "bg-yellow-50 dark:bg-yellow-900/20 text-yellow-700 dark:text-yellow-300"
                      : "bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-300"
                  )}>
                    Calidad de segmentación:&nbsp;<strong>{getDiceLabel(metrics.dice)}</strong>
                    &nbsp;(DICE = {metrics.dice.toFixed(4)})
                  </div>
                </div>
              )} */}
            </div>
          </TabsContent>

          {/* ── Tab: FP / FN ── */}
          {hasFPFN && (
            <TabsContent value="fpfn">
              <div className="space-y-4">
                <FPFNLegend />

                <div className="flex flex-wrap gap-6 justify-center">
                  <ZoomableImage
                    src={`data:image/png;base64,${inferenceResult.fpfn_overlay_b64!}`}
                    alt="Mapa FP/FN"
                    label="Falsos Positivos / Negativos"
                    subtitle="Verde=VP · Rojo=FP · Azul=FN"
                  />
                  {/* MRI con GT superpuesta en lugar de MRI plana */}
                  {hasGT && inferenceResult.gt_overlay_b64 && (
                    <ZoomableImage
                      src={`data:image/png;base64,${inferenceResult.gt_overlay_b64}`}
                      alt="Ground Truth sobre MRI"
                      label="GT sobre MRI (referencia)"
                      badge="Azul = GT"
                    />
                  )}
                </div>

                {metrics && (
                  <div className="rounded-lg border bg-muted/20 p-3 space-y-1.5">
                    <div className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-2">
                      Análisis de errores
                    </div>
                    {[
                      {
                        label: "Verdaderos Positivos (VP)",
                        value: Math.round(
                          (metrics.dice *
                            (metrics.gt_lesion_pixels +
                              metrics.pred_lesion_pixels)) /
                            2,
                        ),
                        color: "text-green-600 dark:text-green-400",
                      },
                      {
                        label: "Falsos Positivos (FP)",
                        value: Math.max(
                          0,
                          metrics.pred_lesion_pixels -
                            Math.round(
                              (metrics.dice *
                                (metrics.gt_lesion_pixels +
                                  metrics.pred_lesion_pixels)) /
                                2,
                            ),
                        ),
                        color: "text-red-600 dark:text-red-400",
                      },
                      {
                        label: "Falsos Negativos (FN)",
                        value: Math.max(
                          0,
                          metrics.gt_lesion_pixels -
                            Math.round(
                              (metrics.dice *
                                (metrics.gt_lesion_pixels +
                                  metrics.pred_lesion_pixels)) /
                                2,
                            ),
                        ),
                        color: "text-blue-600 dark:text-blue-400",
                      },
                    ].map(({ label, value, color }) => (
                      <div key={label} className="flex justify-between text-xs">
                        <span className="text-muted-foreground">{label}</span>
                        <span className={cn("font-mono font-medium", color)}>
                          ~{value.toLocaleString()} px
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </TabsContent>
          )}

          {/* ── Tab: K-Fold ── */}
          <TabsContent value="folds">
            <div className="flex flex-wrap gap-4 justify-center">
              {inferenceResult.fold_masks_b64.map((b64, i) => (
                <div key={i} className="flex flex-col items-center gap-1">
                  <div className="overflow-hidden rounded-md border bg-black/5 dark:bg-white/5">
                    <img
                      src={`data:image/png;base64,${b64}`}
                      alt={`Fold ${i + 1}`}
                      className="mri-image h-28 w-28 object-contain"
                    />
                  </div>
                  <span className="text-xs text-muted-foreground">
                    Fold {i + 1}
                  </span>
                </div>
              ))}
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}
