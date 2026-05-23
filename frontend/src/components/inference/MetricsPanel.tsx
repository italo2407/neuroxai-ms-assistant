import { TrendingUp, Target, BarChart3, Info } from "lucide-react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
  Tooltip,
  TooltipContent,
  TooltipTrigger,
  TooltipProvider,
} from "@/lib/shadcn-components";
import { useAnalysisStore } from "@/store/analysisStore";
import { getDiceColor, getDiceLabel, cn } from "@/lib/utils";

function MetricCard({
  label,
  value,
  color,
  description,
  icon: Icon,
}: {
  label: string;
  value: string;
  color: string;
  description?: string;
  icon: React.ElementType;
}) {
  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <div className="rounded-lg border bg-card p-4 cursor-help hover:shadow-sm transition-shadow">
            <div className="flex items-start justify-between">
              <div className="flex h-8 w-8 items-center justify-center rounded-md bg-muted">
                <Icon className="h-4 w-4 text-muted-foreground" />
              </div>
              <Info className="h-3 w-3 text-muted-foreground/50" />
            </div>
            <div className={cn("mt-2 text-2xl font-bold", color)}>{value}</div>
            <div className="text-xs text-muted-foreground mt-0.5">{label}</div>
          </div>
        </TooltipTrigger>
        {description && (
          <TooltipContent>
            <p className="max-w-xs text-xs">{description}</p>
          </TooltipContent>
        )}
      </Tooltip>
    </TooltipProvider>
  );
}

function PixelBar({
  label,
  value,
  total,
  color,
}: {
  label: string;
  value: number;
  total: number;
  color: string;
}) {
  const pct = total > 0 ? (value / total) * 100 : 0;
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-medium">
          {value.toLocaleString()} px ({pct.toFixed(2)}%)
        </span>
      </div>
      <div className="h-2 w-full rounded-full bg-muted overflow-hidden">
        <div
          className={cn("h-full rounded-full transition-all", color)}
          style={{ width: `${Math.min(pct * 5, 100)}%` }}
        />
      </div>
    </div>
  );
}

export function MetricsPanel() {
  const { metrics, inferenceResult } = useAnalysisStore();

  if (!metrics || !inferenceResult?.has_gt) return null;

  const totalPx = 224 * 224;

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base flex items-center gap-2">
          <span className="flex h-6 w-6 items-center justify-center rounded bg-primary text-primary-foreground text-xs font-bold">
            3
          </span>
          Métricas de Evaluación
        </CardTitle>
        <CardDescription>
          Comparación cuantitativa entre la predicción y el Ground Truth
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <MetricCard
            label="DICE"
            value={metrics.dice.toFixed(4)}
            color={getDiceColor(metrics.dice)}
            icon={Target}
            description="Coeficiente DICE (F1 para segmentación). Rango 0–1. >0.7 = Excelente."
          />
          <MetricCard
            label="IoU / Jaccard"
            value={metrics.iou.toFixed(4)}
            color="text-blue-600 dark:text-blue-400"
            icon={BarChart3}
            description="Intersección sobre la Unión. Más estricto que DICE. IoU = DICE / (2 - DICE)."
          />
          {/* <MetricCard
            label="Precisión"
            value={metrics.precision.toFixed(4)}
            color="text-purple-600 dark:text-purple-400"
            icon={TrendingUp}
            description="VP / (VP + FP). Proporción de píxeles de lesión predichos que son realmente lesión."
          />
          <MetricCard
            label="Recall / Sensibilidad"
            value={metrics.recall.toFixed(4)}
            color="text-orange-600 dark:text-orange-400"
            icon={TrendingUp}
            description="VP / (VP + FN). Proporción de píxeles de lesión real que fueron detectados."
          /> */}
        </div>

        <div
          className={cn(
            "flex items-center gap-2 rounded-lg p-3 text-sm font-medium",
            metrics.dice >= 0.7
              ? "bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-300"
              : metrics.dice >= 0.4
                ? "bg-yellow-50 dark:bg-yellow-900/20 text-yellow-700 dark:text-yellow-300"
                : "bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-300",
          )}
        >
          <Target className="h-4 w-4" />
          Calidad de segmentación: <strong>{getDiceLabel(metrics.dice)}</strong>
          &nbsp;(DICE = {metrics.dice.toFixed(4)})
        </div>

        <div className="space-y-3">
          <div className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
            Área de Lesión
          </div>
          <PixelBar
            label="Ground Truth"
            value={metrics.gt_lesion_pixels}
            total={totalPx}
            color="bg-blue-500"
          />
          <PixelBar
            label="Predicción"
            value={metrics.pred_lesion_pixels}
            total={totalPx}
            color="bg-green-500"
          />
        </div>
      </CardContent>
    </Card>
  );
}
