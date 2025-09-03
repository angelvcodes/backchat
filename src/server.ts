import express from "express";
import cors from "cors";
import fetch from "node-fetch";
import type { Request, Response } from "express";

// 👇 Importar RAG sin extensión
import { loadKnowledge } from "./rag.ts";
import { retrieveContext } from "./retrieveContext.ts";


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
        temperature: 0.2, // 🔽 menos creativo => menos invenciones
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

// ---------------- Utils de embeddings/validación ----------------
const SIM_ACCEPT = 0.70; // ✅ 0.729 pasará
const SIM_BLOCK  = 0.60; // ❌ por debajo, bloqueamos

function cosineSimilarity(vecA: number[], vecB: number[]): number {
  const dot = vecA.reduce((sum, a, i) => sum + a * (vecB[i] || 0), 0);
  const normA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
  const normB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
  const denom = normA * normB || 1e-9;
  return dot / denom;
}

async function getEmbedding(text: string): Promise<number[]> {
  try {
    const resp = await fetch("http://10.0.0.17:1234/v1/embeddings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "nomic-embed-text",
        input: text,
      }),
    });
    const data: any = await resp.json();
    return data?.data?.[0]?.embedding ?? [];
  } catch (err) {
    console.error("❌ Error obteniendo embedding:", err);
    return [];
  }
}

function splitContextChunks(context: string): string[] {
  // asume que tu retrieveContext une chunks con \n\n
  return context
    .split(/\n{2,}/g)
    .map(s => s.trim())
    .filter(s => s.length > 0)
    .slice(0, 6); // límite defensivo
}

// --- Fallback léxico si embeddings fallan ---
function normalizeText(s: string): string {
  return s
    .toLowerCase()
    .normalize("NFD").replace(/[\u0300-\u036f]/g, "") // quita acentos
    .replace(/[^a-z0-9áéíóúüñ\s]/gi, " ")
    .replace(/\s+/g, " ")
    .trim();
}
function tokenSet(s: string): Set<string> {
  const stop = new Set(["la","el","los","las","de","del","y","o","u","a","en","con","para","por","un","una","que","se","es","al","lo"]);
  const toks = normalizeText(s).split(" ").filter(w => w && !stop.has(w) && w.length > 2);
  return new Set(toks);
}
function jaccard(a: Set<string>, b: Set<string>): number {
  let inter = 0;
  for (const t of a) if (b.has(t)) inter++;
  const union = a.size + b.size - inter || 1;
  return inter / union;
}

// ---------------- Validación híbrida (embeddings + léxico) ----------------
async function validateResponseHybrid(respuesta: string, context: string): Promise<string> {
  const cleanResp = respuesta.trim();
  const cleanCtx  = context.trim();

  // Si ya viene el aviso, respetamos
  if (cleanResp.toLowerCase().includes("⚠️ no hay información relevante")) {
    return respuesta;
  }

  // Si no hay contexto, bloqueamos
  if (!cleanCtx) {
    return "⚠️ No hay información relevante en la base de conocimiento.";
  }

  // Embeddings por chunk
  const chunks = splitContextChunks(cleanCtx);
  try {
    const respEmb = await getEmbedding(cleanResp);
    if (respEmb.length) {
      const chunkEmbs = await Promise.all(chunks.map(getEmbedding));
      let maxSim = -1;
      let bestIdx = -1;
      chunkEmbs.forEach((emb, i) => {
        if (!emb.length) return;
        const s = cosineSimilarity(respEmb, emb);
        if (s > maxSim) { maxSim = s; bestIdx = i; }
      });

      console.log(`📊 Similitud máx respuesta↔chunk: ${maxSim.toFixed(3)} (chunk #${bestIdx + 1})`);

      // Reglas suaves
      if (maxSim >= SIM_ACCEPT) {
        return respuesta; // ✅ pasa
      }
      if (maxSim < SIM_BLOCK) {
        return "⚠️ No hay información relevante en la base de conocimiento.";
      }

      // Zona gris: permitimos pero marcamos baja confianza (opcional)
      // Puedes comentar la siguiente línea si no quieres prefijo de advertencia:
      return `ℹ️ (Confianza media, similitud ${maxSim.toFixed(3)})\n${respuesta}`;
    }
  } catch (err) {
    console.error("❌ Error en validación con embeddings:", err);
    // caemos al fallback léxico
  }

  // Fallback léxico (si embeddings fallan)
  const respSet = tokenSet(cleanResp);
  const scores = chunks.map(ch => jaccard(respSet, tokenSet(ch)));
  const best = Math.max(...scores, 0);
  console.log(`🧩 (Fallback) Jaccard máx respuesta↔chunk: ${best.toFixed(3)}`);

  if (best >= 0.20) return respuesta;                 // aceptamos
  if (best < 0.12) return "⚠️ No hay información relevante en la base de conocimiento.";
  return `ℹ️ (Confianza media)\n${respuesta}`;        // zona gris
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
  let respuesta = await generateAIResponse(finalMessages);

  // ✅ Validación post-respuesta (híbrida)
  respuesta = await validateResponseHybrid(respuesta, context);

  // Guardar y enviar respuesta final
  session.messages.push({ role: "assistant", content: respuesta, timestamp: Date.now() });
  res.json({ textResponse: respuesta, contextFound: !respuesta.includes("⚠️") });
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
