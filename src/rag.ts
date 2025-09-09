import fs from "fs";
import path from "path";
import mammoth from "mammoth";
import fetch from "node-fetch";
import { fileURLToPath } from "url";
import dotenv from "dotenv";
dotenv.config();

interface Chunk {
  text: string;
  embedding: number[];
  id?: number;
}

// ----------------------
// ESM-friendly __dirname
// ----------------------
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const EMBEDDING_API_URL = process.env.EMBEDDING_API_URL!;

// Base de conocimiento en memoria
export let knowledgeBase: Chunk[] = [];

// ----------------------
// 1. Leer archivo Word con Mammoth
// ----------------------
export async function loadWordFile(filePath: string): Promise<string> {
  const buffer = fs.readFileSync(filePath);
  const result = await mammoth.extractRawText({ buffer });
  return result.value;
}

// ----------------------
// 2. Dividir texto en chunks (FAQ autosuficientes)
// ----------------------
function chunkFAQs(text: string): string[] {
  return text
    .split(/===\s*.*?\s*===/g)
    .map((c) => c.trim())
    .filter(Boolean);
}

// ----------------------
// 3. Generar embeddings (mejorado)
// ----------------------
interface GetEmbeddingOptions {
  i?: number;
  allowShort?: boolean;
  maxChars?: number;
}

export async function getEmbedding(
  text: string,
  opts?: GetEmbeddingOptions
): Promise<number[] | null> {
  const { i, allowShort = false, maxChars = 4000 } = opts || {};

  try {
    let cleanText = text
      .replace(/[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]/g, "")
      .trim();

    const MIN_CHARS = allowShort ? 1 : 5;
    if (!cleanText || cleanText.length < MIN_CHARS) {
      console.warn(
        `⚠️ Chunk ${
          i !== undefined ? "#" + (i + 1) : ""
        } omitido: texto vacío o demasiado corto (len=${cleanText.length}, min=${MIN_CHARS})`
      );
      return null;
    }

    if (cleanText.length > maxChars) {
      console.warn(
        `⚠️ Chunk ${
          i !== undefined ? "#" + (i + 1) : ""
        } recortado: longitud ${cleanText.length} → ${maxChars}`
      );
      cleanText = cleanText.slice(0, maxChars);
    }

    const response = await fetch(EMBEDDING_API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "text-embedding-granite-embedding-278m-multilingual",
        input: cleanText,
      }),
    });

    const raw = await response.text();

    if (!response.ok) {
      console.error(
        `❌ Error HTTP ${response.status} al pedir embedding para chunk ${
          i !== undefined ? "#" + (i + 1) : ""
        } (len=${cleanText.length}):`,
        raw
      );
      return null;
    }

    let data: any;
    try {
      data = raw ? JSON.parse(raw) : {};
    } catch (parseErr) {
      console.error("❌ No se pudo parsear respuesta JSON:", parseErr, "raw:", raw);
      return null;
    }

    const embedding =
      data?.data?.[0]?.embedding ||
      data?.embedding ||
      data?.vectors?.[0]?.values ||
      null;

    if (!embedding || !Array.isArray(embedding)) {
      console.warn(
        `⚠️ No se recibió embedding válido para chunk ${
          i !== undefined ? "#" + (i + 1) : ""
        }:`,
        JSON.stringify(data).slice(0, 300)
      );
      return null;
    }

    return embedding;
  } catch (err) {
    console.error(
      `❌ Error inesperado en getEmbedding para chunk ${
        opts?.i !== undefined ? "#" + (opts.i + 1) : ""
      }: "${text.slice(0, 80)}..."`,
      err
    );
    return null;
  }
}

// ----------------------
// 4. Similitud coseno segura
// ----------------------
export function cosineSimilarity(a: number[], b: number[]): number {
  const length = Math.min(a.length, b.length);
  if (length === 0) return 0;

  let dot = 0;
  let magA = 0;
  let magB = 0;

  for (let i = 0; i < length; i++) {
    dot += a[i]! * b[i]!;
    magA += a[i]! * a[i]!;
    magB += b[i]! * b[i]!;
  }

  const denominator = Math.sqrt(magA) * Math.sqrt(magB);
  return denominator === 0 ? 0 : dot / denominator;
}

// ----------------------
// 4.1 Detectar saludos
// ----------------------
export function isGreeting(message: string): boolean {
  const normalized = message
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .trim();

  // Lista de palabras/frases que cuentan como saludo
  const keywords = [
    "hola",
    "holaa",
    "buenos dias",
    "buenas tardes",
    "buenas noches",
    "buen dia",
    "que tal",
    "como estas",
    "saludos",
    "hey",
  ];

  return keywords.some((kw) => normalized.includes(kw));
}

// ----------------------
// 5. Cargar base de conocimiento
// ----------------------
export async function loadKnowledge(): Promise<void> {
  const cachePath = path.join(__dirname, "BaseDeConocimiento.json");

  if (fs.existsSync(cachePath)) {
    knowledgeBase = JSON.parse(fs.readFileSync(cachePath, "utf-8"));
    console.log(
      `✅ Base de conocimiento cargada desde cache con ${knowledgeBase.length} chunks`
    );
    return;
  }

  const filePath = path.join(__dirname, "conocimiento.docx");
  const text = await loadWordFile(filePath);
  const chunks = chunkFAQs(text);

  knowledgeBase = [];
  for (const [i, chunk] of chunks.entries()) {
    const embedding = await getEmbedding(chunk, { i });
    if (!embedding) {
      console.warn(
        `⚠️ Chunk #${i + 1} omitido (sin embedding): "${chunk.slice(0, 80)}..."`
      );
      continue;
    }
    knowledgeBase.push({ text: chunk, embedding });
  }

  fs.writeFileSync(cachePath, JSON.stringify(knowledgeBase, null, 2));
  console.log(
    `✅ Base de conocimiento creada y guardada con ${knowledgeBase.length} chunks`
  );
}

// ----------------------
// 6. Recuperar contexto
// ----------------------
export async function retrieveContext(
  question: string,
  topN: number = 2,
  minWords: number = 5,
  minScore: number = 0.65
): Promise<string | null> {
  if (!knowledgeBase.length) return null;

  const qEmbedding = await getEmbedding(question, { allowShort: true });
  if (!qEmbedding) {
    console.warn("⚠️ No se pudo generar embedding para la pregunta.");
    return null;
  }

  const ranked = knowledgeBase
    .map((chunk) => ({
      text: chunk.text,
      score: cosineSimilarity(qEmbedding, chunk.embedding),
    }))
    .sort((a, b) => b.score - a.score);

  console.log(`\n🔎 Resultados de similitud para: "${question}"`);
  ranked.slice(0, 5).forEach((r, i) => {
    console.log(
      `   #${i + 1} → Score: ${r.score.toFixed(3)} | Texto: ${r.text.slice(0, 80)}...`
    );
  });

  const relevant = ranked
    .filter((r) => r.score >= minScore && r.text.split(" ").length >= minWords)
    .slice(0, topN);

  if (relevant.length === 0) {
    console.log("⚠️ No se encontró contexto relevante para esta pregunta.");
    return null;
  }

  console.log(
    `📌 Contexto seleccionado (${relevant.length} chunks sobre threshold ${minScore} y minWords ${minWords}):`
  );
  relevant.forEach((r) =>
    console.log(
      `   → Score: ${r.score.toFixed(3)} | Texto: ${r.text.slice(0, 80)}...`
    )
  );

  return relevant.map((r) => r.text).join("\n\n");
}

// ----------------------
// 7. Guardar preguntas sin respuesta
// ----------------------
const unansweredPath = path.join(__dirname, "new_questions.json");

interface UnansweredEntry {
  message: string;
  timestamp: number;
  context: string[];
  topScore: number;
}

export function saveUnansweredMessage(
  sessionId: string,
  userMessage: string,
  contextFragments: string[],
  topScore: number
) {
  let data: Record<string, UnansweredEntry[]> = {};

  if (fs.existsSync(unansweredPath)) {
    try {
      const raw = fs.readFileSync(unansweredPath, "utf-8");
      data = JSON.parse(raw);
    } catch {
      data = {};
    }
  }

  if (!data[sessionId]) data[sessionId] = [];

  data[sessionId].push({
    message: userMessage,
    timestamp: Date.now(),
    context: contextFragments,
    topScore,
  });

  fs.writeFileSync(unansweredPath, JSON.stringify(data, null, 2), "utf-8");
  console.log(`💾 Mensaje no respondido guardado para session ${sessionId}`);
}

// ----------------------
// 8. Manejar mensajes de usuario
// ----------------------
export async function handleUserMessage(
  sessionId: string,
  userMessage: string
) {
  // 👋 Detectar saludos primero
  if (isGreeting(userMessage)) {
       console.log("✅ Detectado saludo:", userMessage);
    return "👋 ¡Hola! ¿En qué puedo ayudarte hoy?";
  }

  const context = await retrieveContext(userMessage);

  if (!context) {
    saveUnansweredMessage(sessionId, userMessage, [], 0);
    return "⚠️ No hay información relevante en la base de conocimiento.";
  }

  return `📚 Contexto encontrado:\n${context}`;
}
