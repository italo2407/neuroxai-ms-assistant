import { useState } from 'react'
import { chatApi } from '@/services/api'
import { useAnalysisStore } from '@/store/analysisStore'
import type { ChatMessage } from '@/types/api.types'

export function useChat() {
  const [isLoading, setIsLoading] = useState(false)
  const { sessionId, chatMessages, addChatMessage } = useAnalysisStore()

  const sendMessage = async (text: string) => {
    if (!sessionId || !text.trim()) return

    const userMsg: ChatMessage = { role: 'user', content: text }
    addChatMessage(userMsg)
    setIsLoading(true)

    try {
      const history = [...chatMessages, userMsg]
      const response = await chatApi.sendMessage(sessionId, text, history)
      addChatMessage({ role: 'model', content: response.reply })
    } catch (err) {
      addChatMessage({
        role: 'model',
        content: 'Sorry, I could not process your message. Please try again.',
      })
    } finally {
      setIsLoading(false)
    }
  }

  return { sendMessage, isLoading }
}
