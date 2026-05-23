import { useMutation, useQuery } from '@tanstack/react-query'
import { xaiApi } from '@/services/api'
import { useAnalysisStore } from '@/store/analysisStore'

export function useXAI() {
  const { setXAIResult, setStage } = useAnalysisStore()

  return useMutation({
    mutationFn: ({ sessionId, methods }: { sessionId: string; methods: string[] }) =>
      xaiApi.compute(sessionId, methods),
    onMutate: () => setStage('xai_computing'),
    onSuccess: (data) => {
      setXAIResult(data)
      setStage('xai_ready')
    },
    onError: () => setStage('ready'),
  })
}

export function usePrecomputedXAI() {
  const { setXAIResult, setStage } = useAnalysisStore()

  return useMutation({
    mutationFn: (sessionId: string) => xaiApi.loadPrecomputed(sessionId),
    onMutate: () => setStage('xai_computing'),
    onSuccess: (data) => {
      setXAIResult(data)
      setStage('xai_ready')
    },
    onError: () => setStage('ready'),
  })
}

export function usePrecomputedStatus() {
  return useQuery({
    queryKey: ['xai-precomputed-status'],
    queryFn: () => xaiApi.getPrecomputedStatus(),
    staleTime: Infinity,
    retry: false,
  })
}
