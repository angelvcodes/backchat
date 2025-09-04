import express from "express";
import cors from "cors";
import fetch from "node-fetch";
import type { Request, Response } from "express";

// 👇 Importar RAG sin extensión
import { loadKnowledge, retrieveContext } from "./rag.ts";

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

  // Inicializar sesión si no existe
  if (!sessions[sessionId]) sessions[sessionId] = { messages: [], lastActive: Date.now() };
  const session = sessions[sessionId];
  session.lastActive = Date.now();

  // Guardar mensaje del usuario
  session.messages.push({ role: "user", content: message, timestamp: Date.now() });

  // 📝 Imprimir historial completo de la sesión
  console.log(`\n💬 Historial de sesión ${sessionId}:`);
  session.messages.forEach((msg, index) => {
    const time = new Date(msg.timestamp).toLocaleTimeString();
    console.log(`[${sessionId}] [${index + 1}] [${time}] ${msg.role.toUpperCase()}: ${msg.content}`);
  });

  // 🔑 Recuperar contexto del documento
  const context = await retrieveContext(message);

  // ❌ Si no hay contexto relevante, no responder
  if (!context?.trim()) {
    return res.json({
      textResponse: "⚠️ Lo siento, no tengo información en la base de conocimiento. Es posible que tu pregunta no esté redactada de la mejor forma; intenta hacerla más precisa, breve y clara.",
      contextFound: false,
    });
  }

  // Preparar mensajes para el modelo solo con contexto válido
 const finalMessages: ChatMessage[] = [
  {
    role: "system",
    content: `Responde SOLO usando la información del siguiente contexto. 
Si la pregunta del usuario no está cubierta en el contexto, responde exactamente con: 
"⚠️ No hay información relevante en la base de conocimiento." 
No intentes inventar información ni completar con conocimiento general.

Contexto disponible:
${context}`,
    timestamp: Date.now(),
  },
  ...session.messages,
];

  // Generar respuesta del asistente
  const respuesta = await generateAIResponse(finalMessages);
  session.messages.push({ role: "assistant", content: respuesta, timestamp: Date.now() });

  res.json({ textResponse: respuesta, contextFound: true });
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
