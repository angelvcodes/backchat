import express from "express";
import cors from "cors";
import fetch from "node-fetch";
import type { Request, Response } from "express";

// 👇 Importar RAG sin extensión
import { loadKnowledge, retrieveContext } from "./rag.js";

const app = express();
app.use(cors());
app.use(express.json());

// ---------------- Sesiones ----------------
interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: number;
}

interface Session {
  messages: ChatMessage[];
  lastActive: number;
}

const sessions: Record<string, Session> = {};
const SESSION_EXPIRATION = 30 * 60 * 1000; // 30 min

// Limpiar sesiones expiradas
setInterval(() => {
  const now = Date.now();
  for (const id in sessions) {
  const session = sessions[id];
  if (!session) continue; // ✅ evita undefined

  if (Date.now() - session.lastActive > SESSION_EXPIRATION) {
    console.log(`🗑️ Sesión ${id} eliminada por inactividad`);
    delete sessions[id];
  }
}
}, 5 * 60 * 1000);

// ---------------- Respuesta IA ----------------
interface LMChoice {
  message: { role: "assistant" | "user" | "system"; content: string };
}
interface LMResponse {
  choices: LMChoice[];
}

function isLMResponse(data: any): data is LMResponse {
  return Array.isArray(data?.choices) && data.choices.every((c: LMChoice) => !!c?.message?.content);

}

async function generateAIResponse(messages: ChatMessage[]): Promise<string> {
  try {
    const lmResponse = await fetch("http://10.0.0.17:1234/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "llama-3.1-8b-ultralong-1m-instruct",
        messages: messages.map(m => ({ role: m.role, content: m.content })),
        max_tokens: 200,
        temperature: 0.7,
        stream: false,
      }),
    });

    const raw = await lmResponse.json();
    if (!isLMResponse(raw)) return "⚠️ Respuesta inesperada del modelo.";
    return raw.choices[0]?.message?.content ?? "⚠️ Respuesta vacía del modelo.";
  } catch (err) {
    console.error(err);
    return "⚠️ Error al conectar con el modelo de IA.";
  }
}

// ---------------- Endpoint /chat ----------------
app.post("/chat", async (req: Request, res: Response) => {
  const { sessionId, message } = req.body;
  if (!sessionId || !message) return res.status(400).json({ error: "Falta sessionId o mensaje" });

  if (!sessions[sessionId]) sessions[sessionId] = { messages: [], lastActive: Date.now() };
  const session = sessions[sessionId];
  session.lastActive = Date.now();

  session.messages.push({ role: "user", content: message, timestamp: Date.now() });

  // 🔑 Buscar siempre contexto
  const context = await retrieveContext(message);
  const finalMessages: ChatMessage[] = [
    {
      role: "system",
      content: context?.trim()
        ? `Usa este contexto como referencia del documento:\n${context}`
        : "No encontré información relevante en el documento. Responde de manera general con tu conocimiento.",
      timestamp: Date.now(),
    },
    ...session.messages,
  ];

  const respuesta = await generateAIResponse(finalMessages);
  session.messages.push({ role: "assistant", content: respuesta, timestamp: Date.now() });

  res.json({ textResponse: respuesta, contextFound: !!context?.trim() });
});

// ---------------- Endpoint /history ----------------
app.get("/history/:sessionId", (req, res) => {
  const session = sessions[req.params.sessionId];
  if (!session) return res.status(404).json({ error: "Sesión no encontrada" });
  res.json(session.messages);
});

// ---------------- Iniciar servidor ----------------
const PORT = 3001;
app.listen(PORT, async () => {
  console.log(`🚀 Backend listo en http://localhost:${PORT}`);
  await loadKnowledge();
});
