import { Zap, RotateCcw } from "lucide-react";
import { ImageUploadZone } from "./ImageUploadZone";
import {
  Button,
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/lib/shadcn-components";
import { useAnalysisStore } from "@/store/analysisStore";
import { useInference } from "@/hooks/useInference";
import { formatMs } from "@/lib/utils";

export function UploadPanel() {
  const {
    imageFile,
    gtMaskFile,
    imagePreviewUrl,
    gtPreviewUrl,
    setImageFile,
    setGtMaskFile,
    stage,
    inferenceResult,
    reset,
  } = useAnalysisStore();

  const inference = useInference();

  const handleRunInference = () => {
    if (!imageFile) return;
    inference.mutate({ imageFile, gtMaskFile });
  };

  const isLoading = stage === "inferring" || inference.isPending;
  const canInfer = !!imageFile && !isLoading;

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base flex items-center gap-2">
          <span className="flex h-6 w-6 items-center justify-center rounded bg-primary text-primary-foreground text-xs font-bold">
            1
          </span>
          Carga e Inferencia
        </CardTitle>
        <CardDescription>
          Carga una imagen MRI y opcionalmente la máscara de referencia
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <ImageUploadZone
          label="Imagen MRI"
          previewUrl={imagePreviewUrl}
          onFile={(file, url) => setImageFile(file, url)}
          onClear={() => setImageFile(null, null)}
          required
          hint="PNG en escala de grises, corte axial (224×224 recomendado)"
        />

        <ImageUploadZone
          label="Máscara Ground Truth (opcional)"
          previewUrl={gtPreviewUrl}
          onFile={(file, url) => setGtMaskFile(file, url)}
          onClear={() => setGtMaskFile(null, null)}
          hint="Máscara binaria PNG — habilita el cálculo de DICE e IoU"
        />

        <div className="flex gap-2">
          <Button
            className="flex-1"
            onClick={handleRunInference}
            disabled={!canInfer}
          >
            {isLoading ? (
              <>
                <span className="mr-2 inline-block h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
                Ejecutando inferencia…
              </>
            ) : (
              <>
                <Zap className="mr-2 h-4 w-4" />
                Ejecutar Inferencia
              </>
            )}
          </Button>

          {inferenceResult && (
            <Button
              variant="outline"
              size="icon"
              onClick={reset}
              title="Reiniciar"
            >
              <RotateCcw className="h-4 w-4" />
            </Button>
          )}
        </div>

        {inferenceResult && (
          <div className="rounded-md bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 px-3 py-2 text-xs text-green-700 dark:text-green-300">
            ✓ Inferencia completada en{" "}
            {formatMs(inferenceResult.inference_time_ms)} · Ensemble{" "}
            {inferenceResult.fold_masks_b64.length} folds
          </div>
        )}

        {inference.isError && (
          <div className="rounded-md bg-destructive/10 border border-destructive/30 px-3 py-2 text-xs text-destructive">
            Error en la inferencia. Verifica que el backend está activo y los
            modelos están cargados.
          </div>
        )}
      </CardContent>
    </Card>
  );
}
