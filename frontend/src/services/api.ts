import axios from 'axios'
import type {
  InferenceResponse, MetricsResponse, XAIResponse,
  ChatMessage, ChatResponse, VLGCBMResponse, ReportRequest,
  PrecomputedStatusResponse,
} from '@/types/api.types'

const api = axios.create({
  baseURL: '/api/v1',
  timeout: 180000, // 3 min for XAI computation
})

export const inferenceApi = {
  predict: async (imageFile: File, gtMaskFile?: File | null): Promise<InferenceResponse> => {
    const form = new FormData()
    form.append('image_file', imageFile)
    if (gtMaskFile) form.append('gt_mask_file', gtMaskFile)
    const { data } = await api.post<InferenceResponse>('/inference/predict', form, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    return data
  },
}

export const metricsApi = {
  compute: async (sessionId: string): Promise<MetricsResponse> => {
    const { data } = await api.post<MetricsResponse>('/metrics/compute', {
      session_id: sessionId,
    })
    return data
  },
  getSession: async (sessionId: string): Promise<Record<string, number>> => {
    const { data } = await api.get(`/metrics/session/${sessionId}`)
    return data
  },
}

export const xaiApi = {
  getMethods: async (): Promise<string[]> => {
    const { data } = await api.get<{ methods: string[] }>('/xai/methods')
    return data.methods
  },
  compute: async (
    sessionId: string,
    methods: string[],
    colormap: string = 'hot'
  ): Promise<XAIResponse> => {
    const { data } = await api.post<XAIResponse>('/xai/compute', {
      session_id: sessionId,
      methods,
      colormap,
    }, { timeout: 600_000 })  // 10 min: methods run sequentially, SHAP+LIME are slow
    return data
  },
  getPrecomputedStatus: async (): Promise<PrecomputedStatusResponse> => {
    const { data } = await api.get<PrecomputedStatusResponse>('/xai/precomputed/status')
    return data
  },
  loadPrecomputed: async (
    sessionId: string,
    colormap: string = 'hot'
  ): Promise<XAIResponse> => {
    const { data } = await api.post<XAIResponse>('/xai/precomputed', {
      session_id: sessionId,
      colormap,
    })
    return data
  },
}

export const chatApi = {
  sendMessage: async (
    sessionId: string,
    message: string,
    history: ChatMessage[]
  ): Promise<ChatResponse> => {
    const { data } = await api.post<ChatResponse>('/chat/message', {
      session_id: sessionId,
      message,
      history,
    })
    return data
  },
  vlgCbm: async (sessionId: string, xaiMethod: string = 'ensemble_mean'): Promise<VLGCBMResponse> => {
    const { data } = await api.post<VLGCBMResponse>('/chat/vlg-cbm', {
      session_id: sessionId,
      xai_method: xaiMethod,
    })
    return data
  },
}

export const reportApi = {
  generate: async (request: Omit<ReportRequest, 'format' | 'language'>): Promise<Blob> => {
    const { data } = await api.post('/report/generate', request, {
      responseType: 'blob',
    })
    return data as Blob
  },
}

export const healthApi = {
  check: async () => {
    const { data } = await axios.get('/health')
    return data
  },
}
