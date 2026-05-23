import { Brain, Moon, Sun, Activity } from 'lucide-react'
import { Button } from '@/lib/shadcn-components'
import { useTheme } from '@/hooks/useTheme'
import { useAnalysisStore } from '@/store/analysisStore'
import { cn } from '@/lib/utils'

const STAGE_LABELS: Record<string, string> = {
  idle: 'Ready',
  uploading: 'Uploading…',
  inferring: 'Running Inference…',
  ready: 'Analysis Ready',
  xai_computing: 'Computing XAI…',
  xai_ready: 'XAI Complete',
  reporting: 'Generating Report…',
}

export function Header() {
  const { isDark, toggle } = useTheme()
  const { stage } = useAnalysisStore()

  return (
    <header className="sticky top-0 z-40 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container flex h-16 items-center gap-4 px-4 md:px-6">
        {/* Logo */}
        <div className="flex items-center gap-2">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary text-primary-foreground">
            <Brain className="h-5 w-5" />
          </div>
          <div className="hidden sm:block">
            <div className="text-sm font-bold leading-tight">NeuroXAI</div>
            <div className="text-xs text-muted-foreground">MS Assistant</div>
          </div>
        </div>

        {/* Stage indicator */}
        <div className="flex flex-1 items-center gap-2">
          <div className={cn(
            'flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium',
            stage === 'idle' ? 'bg-muted text-muted-foreground' : 'bg-primary/10 text-primary'
          )}>
            {stage !== 'idle' && <Activity className="h-3 w-3 animate-pulse" />}
            {STAGE_LABELS[stage] || stage}
          </div>
        </div>

        {/* Actions */}
        <div className="flex items-center gap-2">
          <span className="hidden md:block text-xs text-muted-foreground font-medium">
            MS Lesion Segmentation · XAI · GenAI
          </span>
          <Button
            variant="ghost"
            size="icon"
            onClick={toggle}
            aria-label="Toggle theme"
          >
            {isDark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
          </Button>
        </div>
      </div>
    </header>
  )
}
