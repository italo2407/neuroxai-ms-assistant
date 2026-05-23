import { type ClassValue, clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function fileToPreviewUrl(file: File): string {
  return URL.createObjectURL(file)
}

export function getDiceColor(dice: number): string {
  if (dice >= 0.7) return 'text-green-600 dark:text-green-400'
  if (dice >= 0.4) return 'text-yellow-600 dark:text-yellow-400'
  return 'text-red-600 dark:text-red-400'
}

export function getDiceLabel(dice: number): string {
  if (dice >= 0.7) return 'Excellent'
  if (dice >= 0.4) return 'Moderate'
  return 'Low'
}

export function formatMs(ms: number): string {
  if (ms < 1000) return `${ms.toFixed(0)}ms`
  return `${(ms / 1000).toFixed(1)}s`
}

export function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}
