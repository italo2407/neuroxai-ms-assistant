import { create } from 'zustand'
import type {
  InferenceResponse, MetricsResponse, XAIResponse, ChatMessage, VLGCBMResponse
} from '@/types/api.types'

export type AppStage =
  | 'idle'
  | 'uploading'
  | 'inferring'
  | 'ready'
  | 'xai_computing'
  | 'xai_ready'
  | 'reporting'

interface AnalysisState {
  // Stage
  stage: AppStage

  // Uploaded files
  imageFile: File | null
  gtMaskFile: File | null
  imagePreviewUrl: string | null
  gtPreviewUrl: string | null

  // Session
  sessionId: string | null

  // Inference results
  inferenceResult: InferenceResponse | null

  // Metrics
  metrics: MetricsResponse | null

  // XAI
  xaiResult: XAIResponse | null
  selectedXAIMethods: string[]

  // Chat
  chatMessages: ChatMessage[]
  isChatOpen: boolean

  // VLG-CBM
  vlgCbmResult: VLGCBMResponse | null

  // Actions
  setImageFile: (file: File | null, previewUrl: string | null) => void
  setGtMaskFile: (file: File | null, previewUrl: string | null) => void
  setStage: (stage: AppStage) => void
  setInferenceResult: (result: InferenceResponse) => void
  setMetrics: (metrics: MetricsResponse | null) => void
  setXAIResult: (result: XAIResponse) => void
  setSelectedXAIMethods: (methods: string[]) => void
  addChatMessage: (msg: ChatMessage) => void
  setChatOpen: (open: boolean) => void
  setVlgCbmResult: (result: VLGCBMResponse) => void
  reset: () => void
}

const DEFAULT_XAI_METHODS = [
  'gradcam', 'integrated_gradients', 'shap', 'lime', 'smoothgrad2'
]

export const useAnalysisStore = create<AnalysisState>((set) => ({
  stage: 'idle',
  imageFile: null,
  gtMaskFile: null,
  imagePreviewUrl: null,
  gtPreviewUrl: null,
  sessionId: null,
  inferenceResult: null,
  metrics: null,
  xaiResult: null,
  selectedXAIMethods: DEFAULT_XAI_METHODS,
  chatMessages: [],
  isChatOpen: false,
  vlgCbmResult: null,

  setImageFile: (file, previewUrl) => set({ imageFile: file, imagePreviewUrl: previewUrl }),
  setGtMaskFile: (file, previewUrl) => set({ gtMaskFile: file, gtPreviewUrl: previewUrl }),
  setStage: (stage) => set({ stage }),
  setInferenceResult: (result) => set({
    inferenceResult: result,
    sessionId: result.session_id,
    stage: 'ready',
  }),
  setMetrics: (metrics) => set({ metrics }),
  setXAIResult: (result) => set({ xaiResult: result, stage: 'xai_ready' }),
  setSelectedXAIMethods: (methods) => set({ selectedXAIMethods: methods }),
  addChatMessage: (msg) => set((s) => ({ chatMessages: [...s.chatMessages, msg] })),
  setChatOpen: (open) => set({ isChatOpen: open }),
  setVlgCbmResult: (result) => set({ vlgCbmResult: result }),
  reset: () => set({
    stage: 'idle',
    imageFile: null,
    gtMaskFile: null,
    imagePreviewUrl: null,
    gtPreviewUrl: null,
    sessionId: null,
    inferenceResult: null,
    metrics: null,
    xaiResult: null,
    chatMessages: [],
    vlgCbmResult: null,
  }),
}))
