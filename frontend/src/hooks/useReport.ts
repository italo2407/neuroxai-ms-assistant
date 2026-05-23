import { useState } from 'react'
import { reportApi } from '@/services/api'
import { useAnalysisStore } from '@/store/analysisStore'

export function useReport() {
  const [isGenerating, setIsGenerating] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const { sessionId } = useAnalysisStore()

  const generate = async (options: { include_xai_maps: boolean; include_vlg_cbm: boolean; patient_label?: string }) => {
    if (!sessionId) return
    setIsGenerating(true)
    setError(null)

    try {
      const blob = await reportApi.generate({ session_id: sessionId, ...options })
      // Open HTML report in a new browser tab. The user can print to PDF via Ctrl+P.
      const url = URL.createObjectURL(blob)
      window.open(url, '_blank')
      setTimeout(() => URL.revokeObjectURL(url), 60_000)
    } catch {
      setError('Error al generar el informe. Inténtalo de nuevo.')
    } finally {
      setIsGenerating(false)
    }
  }

  return { generate, isGenerating, error }
}
