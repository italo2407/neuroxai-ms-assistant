import { useMutation } from '@tanstack/react-query'
import { inferenceApi } from '@/services/api'
import { useAnalysisStore } from '@/store/analysisStore'

export function useInference() {
  const { setInferenceResult, setMetrics, setStage } = useAnalysisStore()

  return useMutation({
    mutationFn: ({ imageFile, gtMaskFile }: { imageFile: File; gtMaskFile?: File | null }) =>
      inferenceApi.predict(imageFile, gtMaskFile),
    onMutate: () => setStage('inferring'),
    onSuccess: (data) => {
      setInferenceResult(data)
      if (data.metrics) setMetrics(data.metrics)
      setStage('ready')
    },
    onError: () => setStage('idle'),
  })
}
