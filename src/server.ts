import express from "express";
import cors from "cors";
import fetch from "node-fetch";
import type { Request, Response } from "express";
import { saveUnansweredMessage } from "./rag.ts";
import { isGreeting } from "./rag.ts";
import dotenv from "dotenv";
dotenv.config();



// Importar RAG sin extensión
import {
  loadKnowledge,
  retrieveContext,
  getEmbedding,
  cosineSimilarity,
  knowledgeBase,
} from "./rag.ts";

const app = express();
app.use(cors());
app.use(express.json());
const LLM_API_URL = process.env.LLM_API_URL!;
const PORT = process.env.PORT!;


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
    if (!session) continue; 

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
  return (
    Array.isArray(data?.choices) &&
    data.choices.every((c: LMChoice) => !!c?.message?.content)
  );
}

async function generateAIResponse(messages: ChatMessage[]): Promise<string> {
  try {
    const lmResponse = await fetch(
      LLM_API_URL,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "llama-3.1-8b-ultralong-1m-instruct",
          messages: messages.map((m) => ({ role: m.role, content: m.content })),
          max_tokens: 200,
          temperature: 0.7,
          stream: false,
        }),
      }
    );

    const raw = await lmResponse.json();
    if (!isLMResponse(raw)) return "⚠️ Respuesta inesperada del modelo.";
    return raw.choices[0]?.message?.content ?? "⚠️ Respuesta vacía del modelo.";
  } catch (err) {
    console.error(err);
    return "⚠️ Error al conectar con el modelo de IA.";
  }
}

// ------------------ Endpoint /chat mejorado ------------------
app.post("/chat", async (req: Request, res: Response) => {
  const { sessionId, message } = req.body;
  if (!sessionId || !message)
    return res.status(400).json({ error: "Falta sessionId o mensaje" });

  // ------------------ Validación de mensaje flexible ------------------
  function isValidMessage(msg: string): boolean {
    const trimmed = msg.trim();

    // Bloquear mensajes muy cortos (menos de 5 caracteres)
    if (trimmed.length < 5) return false;

    // Debe contener al menos una letra
    if (!/[a-zA-Z]/.test(trimmed)) return false;

    // Bloquear mensajes que sean solo símbolos o números
    if (/^[^a-zA-Z]+$/.test(trimmed)) return false;

    return true;
  }

  if (!isValidMessage(message)) {
    const invalidMsg =
      "⚠️ No entiendo tu mensaje. Por favor escribe una oración o pregunta clara.";
    return res.json({ textResponse: invalidMsg, contextFound: false });
  }

  // ------------------ Manejo de sesión ------------------
  if (!sessions[sessionId])
    sessions[sessionId] = { messages: [], lastActive: Date.now() };
  const session = sessions[sessionId];
  session.lastActive = Date.now();

  session.messages.push({
    role: "user",
    content: message,
    timestamp: Date.now(),
  });

  // ------------------ Detección de saludo ------------------
  if (isGreeting(message)) {
    const saludo = "👋 ¡Hola! ¿En qué puedo ayudarte hoy?";
    session.messages.push({
      role: "assistant",
      content: saludo,
      timestamp: Date.now(),
    });
    return res.json({ textResponse: saludo, contextFound: false });
  }

  // ------------------ Recuperar contexto ------------------
  const context = await retrieveContext(message, 2, 5, 0.65);
  const qEmbedding = await getEmbedding(message);
  let topScore = 0;

  if (qEmbedding) {
    topScore =
      knowledgeBase
        .map((c) => cosineSimilarity(qEmbedding, c.embedding))
        .sort((a, b) => b - a)[0] ?? 0;
  } else {
    console.warn(`⚠️ No se pudo generar embedding para la pregunta: "${message}"`);
  }

  const STRICT_THRESHOLD = 0.7;

  // ------------------ Manejo de excepción si no hay contexto suficiente ------------------
  if (!context?.trim() || topScore < STRICT_THRESHOLD) {
    const warningMessage =
      "⚠️ Aun estoy aprendiendo y no tengo la respuesta, pero la guardaré para que los responsables la revisen pronto.";

    saveUnansweredMessage(sessionId, message, context ? [context] : [], topScore);

    session.messages.push({
      role: "assistant",
      content: warningMessage,
      timestamp: Date.now(),
    });

    return res.json({
      textResponse: warningMessage,
      contextFound: false,
    });
  }

  // ------------------ Preparar prompt para LLM ------------------
  const systemPrompt = `
Eres un asistente virtual de la Alcaldía de Yopal.  
Responde SOLO con la información del contexto.  

Reglas:  
- Si la pregunta no está cubierta, responde:
  "⚠️ No hay información relevante en la base de conocimiento."  
- No inventes información ni uses conocimiento externo.  
- Reescribe las respuestas en un tono claro, cordial y natural, no la copies textualmente.
---
📚 Contexto:
${context}
`;

  const lastMessages = session.messages.slice(-1);
  const finalMessages: ChatMessage[] = [
    { role: "system", content: systemPrompt, timestamp: Date.now() },
    ...lastMessages,
  ];

  const respuesta = await generateAIResponse(finalMessages);

  // Guardar respuesta del asistente
  session.messages.push({
    role: "assistant",
    content: respuesta,
    timestamp: Date.now(),
  });

  // ------------------ Logging de chunks más relevantes ------------------
  let ranked: { text: string; score: number }[] = [];

  if (qEmbedding) {
    ranked = knowledgeBase
      .map((chunk) => ({
        text: chunk.text,
        score: cosineSimilarity(qEmbedding, chunk.embedding),
      }))
      .sort((a, b) => b.score - a.score);
  } else {
    console.warn("⚠️ No se pudo generar embedding para la pregunta, no se calcularán scores.");
  }
  ranked.slice(0, 3).forEach((r, i) =>
    console.log(
      `   #${i + 1} → Score: ${r.score.toFixed(3)} | Texto: ${r.text.slice(0, 80)}...`
    )
  );

  return res.json({ textResponse: respuesta, contextFound: true });
});

// ---------------- Iniciar servidor ----------------

app.listen(PORT, async () => {
  console.log(`🚀 Backend listo en http://localhost:${PORT}`);
  //arranca la funcion para crear los chucks
  await loadKnowledge();
});
