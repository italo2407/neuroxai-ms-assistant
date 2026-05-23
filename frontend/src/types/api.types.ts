export interface InferenceResponse {
  session_id: string
  predicted_mask_b64: string
  soft_logits_b64: string
  overlay_b64: string
  gt_overlay_b64?: string | null
  fpfn_overlay_b64?: string | null
  fold_masks_b64: string[]
  inference_time_ms: number
  image_size: number[]
  has_gt: boolean
  metrics?: MetricsResponse | null
}

export interface MetricsResponse {
  dice: number
  iou: number
  gt_lesion_pixels: number
  pred_lesion_pixels: number
  gt_lesion_pct: number
  pred_lesion_pct: number
  precision: number
  recall: number
  f1: number
}

export interface XAIMapResult {
  heatmap_b64: string
  heatmap_overlay_b64: string
  iou_vs_gt: number | null
  timed_out: boolean
}

export interface XAIResponse {
  session_id: string
  maps: Record<string, XAIMapResult>
  compute_time_ms: number
  vlg_cbm_concepts: Record<string, number | string>
  methods_available: string[]
}

export interface ChatMessage {
  role: 'user' | 'model'
  content: string
}

export interface ChatResponse {
  reply: string
  role: string
}

export interface VLGCBMResponse {
  concepts: Record<string, number | string>
  clinical_explanation: string
  model_used: string
}

export interface PrecomputedStatusResponse {
  available: boolean
  methods: string[]
  source_dir: string | null
}

export interface ReportRequest {
  session_id: string
  include_xai_maps: boolean
  include_vlg_cbm: boolean
  patient_label?: string
}

export const XAI_METHOD_LABELS: Record<string, string> = {
  gradcam: 'Grad-CAM',
  integrated_gradients: 'Gradientes Integrados',
  shap: 'SHAP',
  lime: 'LIME',
  smoothgrad2: 'SmoothGrad²',
}

export const XAI_METHOD_COLORS: Record<string, string> = {
  gradcam: 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300',
  integrated_gradients: 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300',
  shap: 'bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-300',
  lime: 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-300',
  smoothgrad2: 'bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-300',
}
