import { useState, useRef, useEffect } from 'react'
import { MessageCircle, X, Send, Bot, User, Brain } from 'lucide-react'
import { Button, Textarea, ScrollArea, Badge } from '@/lib/shadcn-components'
import { useAnalysisStore } from '@/store/analysisStore'
import { useChat } from '@/hooks/useChat'
import { cn } from '@/lib/utils'
import ReactMarkdown from 'react-markdown'

function MessageBubble({ role, content }: { role: string; content: string }) {
  const isUser = role === 'user'
  return (
    <div className={cn('flex gap-2 text-sm', isUser ? 'flex-row-reverse' : 'flex-row')}>
      <div className={cn(
        'flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-xs font-bold',
        isUser ? 'bg-primary text-primary-foreground' : 'bg-muted text-muted-foreground'
      )}>
        {isUser ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
      </div>
      <div className={cn(
        'rounded-2xl px-4 py-2.5 max-w-[85%] shadow-sm',
        isUser
          ? 'bg-primary text-primary-foreground rounded-tr-sm'
          : 'bg-muted text-foreground rounded-tl-sm'
      )}>
        <ReactMarkdown className="prose prose-sm dark:prose-invert max-w-none text-inherit">
          {content}
        </ReactMarkdown>
      </div>
    </div>
  )
}

export function FloatingChat() {
  const { isChatOpen, setChatOpen, chatMessages, sessionId } = useAnalysisStore()
  const { sendMessage, isLoading } = useChat()
  const [input, setInput] = useState('')
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [chatMessages, isLoading])

  const handleSend = () => {
    if (!input.trim() || isLoading) return
    sendMessage(input.trim())
    setInput('')
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <>
      {/* FAB */}
      {!isChatOpen && (
        <button
          onClick={() => setChatOpen(true)}
          className={cn(
            'fixed bottom-6 right-6 z-50 flex h-14 w-14 items-center justify-center rounded-full shadow-lg',
            'bg-primary text-primary-foreground hover:bg-primary/90 transition-all',
            'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
            'animate-fade-in'
          )}
          aria-label="Abrir chat clínico"
        >
          <Brain className="h-6 w-6" />
          {chatMessages.length > 0 && (
            <span className="absolute -top-1 -right-1 flex h-5 w-5 items-center justify-center rounded-full bg-destructive text-[10px] font-bold text-destructive-foreground">
              {chatMessages.length > 9 ? '9+' : chatMessages.length}
            </span>
          )}
        </button>
      )}

      {/* Chat panel */}
      {isChatOpen && (
        <div className={cn(
          'fixed bottom-6 right-6 z-50 flex flex-col rounded-2xl border shadow-2xl',
          'w-[380px] max-w-[calc(100vw-48px)] bg-background',
          'animate-fade-in',
          'h-[560px] max-h-[calc(100vh-96px)]'
        )}>
          {/* Header */}
          <div className="flex items-center gap-3 rounded-t-2xl border-b bg-muted/30 px-4 py-3">
            <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary text-primary-foreground">
              <Brain className="h-4 w-4" />
            </div>
            <div className="flex-1 min-w-0">
              <div className="text-sm font-semibold">NeuroXAI Assistant</div>
              <div className="text-[10px] text-muted-foreground">
                {sessionId ? 'Sesión activa · Gemini' : 'Sin sesión — ejecuta la inferencia primero'}
              </div>
            </div>
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7 shrink-0"
              onClick={() => setChatOpen(false)}
            >
              <X className="h-4 w-4" />
            </Button>
          </div>

          {/* Messages */}
          <ScrollArea className="flex-1 px-4">
            <div className="space-y-4 py-4">
              {chatMessages.length === 0 && (
                <div className="text-center py-8 space-y-3">
                  <Brain className="h-10 w-10 mx-auto text-muted-foreground/30" />
                  <div className="text-sm text-muted-foreground">
                    Pregúntame sobre la segmentación de lesiones EM, los mapas XAI o la interpretación clínica.
                  </div>
                  <div className="flex flex-wrap gap-2 justify-center">
                    {[
                      'Explica el coeficiente DICE',
                      '¿Qué sugieren las lesiones periventriculares?',
                      'Interpreta el mapa Grad-CAM',
                    ].map((s) => (
                      <button
                        key={s}
                        onClick={() => { setInput(s); }}
                        className="rounded-full border px-3 py-1 text-xs text-muted-foreground hover:bg-muted/50 transition-colors"
                      >
                        {s}
                      </button>
                    ))}
                  </div>
                </div>
              )}
              {chatMessages.map((msg, i) => (
                <MessageBubble key={i} role={msg.role} content={msg.content} />
              ))}
              {isLoading && (
                <div className="flex gap-2 items-center">
                  <div className="flex h-7 w-7 items-center justify-center rounded-full bg-muted">
                    <Bot className="h-4 w-4 text-muted-foreground" />
                  </div>
                  <div className="flex gap-1 rounded-2xl rounded-tl-sm bg-muted px-4 py-3">
                    {[0, 1, 2].map((i) => (
                      <span key={i} className="inline-block h-2 w-2 rounded-full bg-muted-foreground/50 animate-bounce" style={{ animationDelay: `${i * 0.15}s` }} />
                    ))}
                  </div>
                </div>
              )}
              <div ref={bottomRef} />
            </div>
          </ScrollArea>

          {/* Input */}
          <div className="border-t p-3">
            <div className="flex gap-2">
              <Textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Pregunta sobre el análisis…"
                className="min-h-[40px] max-h-24 resize-none text-sm py-2"
                disabled={!sessionId || isLoading}
              />
              <Button
                size="icon"
                onClick={handleSend}
                disabled={!input.trim() || !sessionId || isLoading}
                className="shrink-0 self-end"
              >
                <Send className="h-4 w-4" />
              </Button>
            </div>
            <div className="mt-1 text-[10px] text-muted-foreground text-center">
              Enter para enviar · Shift+Enter para nueva línea · Powered by Gemini
            </div>
          </div>
        </div>
      )}
    </>
  )
}
