import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Header } from '@/components/layout/Header'
import { AnalysisPage } from '@/pages/AnalysisPage'
import { FloatingChat } from '@/components/chat/FloatingChat'
import { useTheme } from '@/hooks/useTheme'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { retry: 1 },
    mutations: { retry: 0 },
  },
})

function AppInner() {
  // Initialize theme from localStorage/system preference
  useTheme()

  return (
    <div className="min-h-screen bg-background font-sans antialiased">
      <Header />
      <AnalysisPage />
      <FloatingChat />
    </div>
  )
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AppInner />
    </QueryClientProvider>
  )
}
