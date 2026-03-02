"use client";

import { useState, useEffect, useRef } from "react";
import { Send, Bot, User, Loader2, BarChart3, TrendingUp, PieChart, Activity, Cpu, Calendar } from "lucide-react";

// --- Configuration ---
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

// --- Types ---
interface Message {
  role: string;
  content: string;
  confidence?: string; // Changed to string to match "85.00%" format from backend
}

interface AnalysisData {
  avgTrades?: string;
  avgValue?: string;
  correlation?: number;
  totalDays?: number;
  charts?: { title: string; url: string }[]; // Optional to prevent .map() crash
}

export default function ChatPage() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [analysis, setAnalysis] = useState<AnalysisData | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetch(`${API_BASE_URL}/`) // Using root endpoint for initial status
      .then((res) => res.json())
      .then((data) => {
        // Fallback for analysis data if separate endpoint isn't ready
        setAnalysis({
          avgTrades: "244_ROWS",
          avgValue: "LOCAL_SYNC",
          correlation: 0.98,
          charts: [
            { title: "MARKET_TREND", url: "/market_trend.png" },
            { title: "VOL_CORRELATION", url: "/correlation_heatmap.png" }
          ]
        });
      })
      .catch((err) => console.error("API Link Error:", err));
  }, []);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isLoading]);

  const sendMessage = async () => {
    if (!input.trim()) return;
    const userMessage = { role: "user", content: input };
    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);
    setInput("");

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: input }),
      });
      const data = await response.json();
      // Updated to match your new FastAPI keys: 'forecast' and 'confidence'
      setMessages((prev) => [...prev, { 
        role: "bot", 
        content: data.forecast, 
        confidence: data.confidence 
      }]);
    } catch (error) {
      setMessages((prev) => [...prev, { role: "bot", content: "❌ ERROR: BACKEND_UNREACHABLE" }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex h-screen bg-[#0D1117] text-slate-300 font-mono selection:bg-[#D2FF00]/30 overflow-hidden">
      
      {/* --- LEFT SIDEBAR: QUANT ANALYTICS --- */}
      <aside className="hidden lg:flex flex-col w-[400px] border-r border-white/5 bg-[#0a0c10] p-5 overflow-y-auto space-y-6">
        <div className="flex items-center gap-2 mb-2">
          <Cpu className="w-4 h-4 text-[#D2FF00]" />
          <h2 className="text-[11px] font-bold text-slate-500 uppercase tracking-[0.2em]">Telemetry_Monitor</h2>
        </div>

        <div className="grid grid-cols-1 gap-3">
          <StatSmall title="MODEL_STATE" value={analysis?.avgTrades || "SYNCED"} icon={<BarChart3 />} />
          <StatSmall title="VOCAB_SIZE" value="42_TOKENS" icon={<TrendingUp />} />
          <StatSmall title="R_COEFFICIENT" value={analysis?.correlation || "0.98"} icon={<PieChart />} />
        </div>

        <div className="pt-4 border-t border-white/5">
          <h2 className="text-[11px] font-bold text-slate-500 uppercase tracking-[0.2em] mb-4">Signal_Visualizer</h2>
          <div className="space-y-6">
            {/* FIX: Double optional chaining prevents the "undefined reading map" error */}
            {analysis?.charts?.map((chart, idx) => {
              const chartUrl = chart.url.startsWith('http') 
                ? chart.url 
                : `${API_BASE_URL}${chart.url}`;
              
              return (
                <TerminalWindow key={idx} title={chart.title}>
                  <img 
                    src={`${chartUrl}?t=${new Date().getTime()}`} 
                    alt={chart.title} 
                    className="w-full grayscale hover:grayscale-0 transition-all duration-700 opacity-80"
                  />
                </TerminalWindow>
              );
            })}
          </div>
        </div>
      </aside>

      {/* --- MAIN CHAT ENGINE --- */}
      <div className="flex-1 flex flex-col relative">
        <header className="px-6 py-4 border-b border-white/5 bg-[#0D1117]/80 backdrop-blur-xl flex items-center justify-between z-10">
          <div className="flex items-center gap-4">
            <div className="p-2.5 bg-[#D2FF00]/10 rounded-lg border border-[#D2FF00]/20">
              <Bot className="text-[#D2FF00] w-6 h-6" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-lg font-black text-white tracking-tighter italic">MARKET_GPT</h1>
                <span className="text-[9px] bg-white/5 px-2 py-0.5 rounded text-slate-500 font-bold">STABLE_V2.0</span>
              </div>
              <p className="text-[10px] text-slate-500 uppercase tracking-widest mt-0.5">Transformer_Core_v3_Flash</p>
            </div>
          </div>
          <div className="flex items-center gap-2 px-4 py-1.5 bg-[#D2FF00]/5 rounded-md border border-[#D2FF00]/10">
            <div className="w-1.5 h-1.5 bg-[#D2FF00] rounded-full animate-ping"></div>
            <span className="text-[10px] text-[#D2FF00] font-black uppercase tracking-tighter">Live_Predictor</span>
          </div>
        </header>

        <main ref={scrollRef} className="flex-1 overflow-y-auto p-6 md:p-10 space-y-8 scrollbar-hide">
          {messages.length === 0 && (
            <div className="h-full flex flex-col items-center justify-center opacity-20">
              <Activity className="w-16 h-16 text-slate-500 mb-4 animate-pulse" />
              <p className="text-xs tracking-[0.3em] font-bold">READY_FOR_MARKET_SEED</p>
            </div>
          )}

          {messages.map((msg, idx) => (
            <div key={idx} className={`flex gap-6 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`max-w-[85%] md:max-w-[70%] p-6 rounded-lg transition-all ${
                msg.role === 'user' 
                  ? 'bg-[#1a1f26] border-r-4 border-[#D2FF00] text-white' 
                  : 'bg-[#0a0c10] border border-white/5 text-slate-200 shadow-2xl'
              }`}>
                <div className="flex items-center gap-2 mb-3 text-[9px] font-bold uppercase tracking-widest text-slate-500">
                  {msg.role === 'user' ? <User className="w-3 h-3" /> : <Bot className="w-3 h-3 text-[#D2FF00]" />}
                  {msg.role === 'user' ? 'Local_Host' : 'Quantum_Predictor'}
                </div>
                <p className="text-sm md:text-base leading-relaxed font-sans font-medium">
                  {msg.content}
                </p>
                {msg.role === 'bot' && msg.confidence && (
                  <ConfidenceMeter score={msg.confidence} />
                )}
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="flex gap-4 items-center opacity-50 italic text-xs tracking-widest">
              <Loader2 className="w-4 h-4 animate-spin text-[#D2FF00]" />
              THINKING_IN_LATENCY...
            </div>
          )}
        </main>

        <footer className="p-8 bg-[#0D1117] border-t border-white/5">
          <div className="max-w-4xl mx-auto relative group">
            <div className="absolute inset-0 bg-[#D2FF00]/5 blur-xl group-focus-within:bg-[#D2FF00]/10 transition-all"></div>
            <div className="relative flex gap-2">
              <div className="flex-1 relative">
                <Calendar className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-600" />
                <input
                  className="w-full bg-black border border-white/10 rounded-lg pl-12 pr-6 py-4 text-sm text-white focus:outline-none focus:border-[#D2FF00]/50 transition-all"
                  placeholder="INPUT_SEED: Start typing market data..."
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && sendMessage()}
                />
              </div>
              <button 
                onClick={sendMessage}
                disabled={isLoading}
                className="bg-[#D2FF00] hover:bg-[#b8e600] disabled:opacity-50 text-black px-6 rounded-lg transition-all active:scale-95 flex items-center justify-center"
              >
                <Send className="w-5 h-5" />
              </button>
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
}

// --- HELPER COMPONENTS ---

function StatSmall({ title, value, icon }: any) {
  return (
    <div className="p-5 bg-white/[0.02] border border-white/[0.05] rounded-lg hover:border-[#D2FF00]/30 transition-all group">
      <div className="flex justify-between items-start mb-3">
        <span className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">{title}</span>
        <div className="text-slate-600 group-hover:text-[#D2FF00] transition-colors">{icon}</div>
      </div>
      <p className="text-2xl font-black text-white italic tracking-tighter">{value}</p>
    </div>
  );
}

function TerminalWindow({ title, children }: any) {
  return (
    <div className="rounded-lg border border-white/10 bg-black/40 overflow-hidden shadow-2xl">
      <div className="flex items-center justify-between px-3 py-2 bg-white/5 border-b border-white/10">
        <div className="flex gap-1.5">
          <div className="w-2 h-2 rounded-full bg-red-500/30"></div>
          <div className="w-2 h-2 rounded-full bg-amber-500/30"></div>
          <div className="w-2 h-2 rounded-full bg-emerald-500/30"></div>
        </div>
        <span className="text-[8px] font-bold text-slate-500 uppercase tracking-[0.2em]">{title}</span>
      </div>
      <div className="p-1">{children}</div>
    </div>
  );
}

function ConfidenceMeter({ score }: { score: string }) {
  const numericScore = parseFloat(score.replace('%', ''));
  return (
    <div className="mt-4 pt-4 border-t border-white/5">
      <div className="flex justify-between items-end mb-2">
        <span className="text-[9px] font-black text-slate-500 uppercase italic">Prediction_Weight</span>
        <span className="text-[12px] font-black text-[#D2FF00]">{score}</span>
      </div>
      <div className="h-1 w-full bg-white/5 rounded-full overflow-hidden">
        <div className="h-full bg-[#D2FF00] shadow-[0_0_10px_#D2FF00]" style={{ width: `${numericScore}%` }} />
      </div>
    </div>
  );
}