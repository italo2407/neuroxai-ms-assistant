import { useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, ImageIcon, X } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Button } from '@/lib/shadcn-components'

interface Props {
  label: string
  previewUrl: string | null
  onFile: (file: File, previewUrl: string) => void
  onClear: () => void
  required?: boolean
  hint?: string
}

export function ImageUploadZone({ label, previewUrl, onFile, onClear, required, hint }: Props) {
  const onDrop = useCallback((accepted: File[]) => {
    if (!accepted[0]) return
    const url = URL.createObjectURL(accepted[0])
    onFile(accepted[0], url)
  }, [onFile])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/png': ['.png'], 'image/jpeg': ['.jpg', '.jpeg'] },
    multiple: false,
    maxSize: 10 * 1024 * 1024,
  })

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <label className="text-sm font-medium text-foreground">
          {label}
          {required && <span className="ml-1 text-destructive">*</span>}
        </label>
        {previewUrl && (
          <Button variant="ghost" size="sm" onClick={onClear} className="h-6 px-2 text-xs">
            <X className="h-3 w-3 mr-1" />
            Quitar
          </Button>
        )}
      </div>

      {previewUrl ? (
        <div className="relative overflow-hidden rounded-lg border bg-muted/20">
          <img
            src={previewUrl}
            alt={label}
            className="mri-image mx-auto block max-h-48 w-full object-contain p-2"
          />
          <div className="absolute bottom-0 left-0 right-0 bg-black/50 py-1 px-2 text-xs text-white text-center">
            {label} cargado
          </div>
        </div>
      ) : (
        <div
          {...getRootProps()}
          className={cn(
            'flex flex-col items-center justify-center rounded-lg border-2 border-dashed',
            'cursor-pointer transition-colors p-8 gap-3',
            isDragActive
              ? 'border-primary bg-primary/5'
              : 'border-border hover:border-primary/50 hover:bg-muted/30'
          )}
        >
          <input {...getInputProps()} />
          <div className="flex h-12 w-12 items-center justify-center rounded-full bg-muted">
            {isDragActive ? (
              <Upload className="h-6 w-6 text-primary animate-bounce" />
            ) : (
              <ImageIcon className="h-6 w-6 text-muted-foreground" />
            )}
          </div>
          <div className="text-center">
            <p className="text-sm font-medium">
              {isDragActive ? 'Suelta la imagen aquí' : 'Arrastra o haz clic para cargar'}
            </p>
            <p className="mt-1 text-xs text-muted-foreground">
              {hint || 'PNG, JPG hasta 10 MB'}
            </p>
          </div>
        </div>
      )}
    </div>
  )
}
