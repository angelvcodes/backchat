import express from "express";
import cors from "cors";
import fetch from "node-fetch";
import type { Request, Response } from "express";

const app = express();
app.use(cors());
app.use(express.json());

// ---------------- Sesiones ----------------
interface ChatMessage {
  role: "user" | "assistant";
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
const cleanExpiredSessions = () => {
  const now = Date.now();
  for (const sessionId in sessions) {
    const session = sessions[sessionId];
    if (!session) continue;

    if (now - session.lastActive > SESSION_EXPIRATION) {
      console.log(`🗑️ Sesión ${sessionId} eliminada por inactividad`);
      delete sessions[sessionId];
    }
  }
};
setInterval(cleanExpiredSessions, 5 * 60 * 1000);

// ---------------- Función de respuesta IA ----------------
async function generateAIResponse(messages: ChatMessage[]): Promise<string> {
  try {
    if (!messages.length) return "⚠️ No hay mensajes para responder.";

    // LM Studio espera un array de mensajes
    const lmResponse = await fetch("http://localhost:1234/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "gpt4all", // ajusta según tu modelo
        messages: messages.map(m => ({ role: m.role, content: m.content })),
        max_new_tokens: 200,
        temperature: 0.7
      })
    });

    const data: unknown = await lmResponse.json();

    if (typeof data === "object" && data !== null && "text" in data) {
      return (data as { text?: string }).text || "Lo siento, no pude generar respuesta.";
    }

    return "Lo siento, no pude generar respuesta.";
  } catch (err) {
    console.error("❌ Error al generar respuesta:", err);
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

  const respuesta = await generateAIResponse(session.messages);

  session.messages.push({ role: "assistant", content: respuesta, timestamp: Date.now() });

  console.log(`\n📌 Conversación [${sessionId}] (actualizada)`);
  session.messages.forEach((msg, i) => {
    console.log(`${i + 1}. ${msg.role.toUpperCase()}: ${msg.content}`);
  });

  res.json({ textResponse: respuesta });
});

// ---------------- Endpoint /history ----------------
app.get("/history/:sessionId", (req, res) => {
  const sessionId = req.params.sessionId;
  if (!sessionId) return res.status(400).json({ error: "Falta sessionId" });

  const session = sessions[sessionId];
  if (!session) return res.status(404).json({ error: "Sesión no encontrada" });

  res.json(session.messages);
});

// ---------------- Iniciar servidor ----------------
const PORT = 3001;
app.listen(PORT, () => console.log(`🚀 Backend listo en http://localhost:${PORT}`));
