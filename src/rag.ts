import fs from "fs";
import path from "path";
import mammoth from "mammoth";
import fetch from "node-fetch";
import { fileURLToPath } from "url";

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
// Cada FAQ debe estar delimitado en el documento con ===
function chunkFAQs(text: string): string[] {
  return text
    .split(/\n===.*?===\n/g)
    .map((c) => c.trim())
    .filter(Boolean);
}

// ----------------------
// 3. Generar embeddings
// ----------------------

interface EmbeddingResponse {
  object: string;
  data: { embedding: number[]; index: number }[];
  model?: string;
  usage?: { prompt_tokens: number; total_tokens: number };
}

export async function getEmbedding(text: string, i?: number): Promise<number[] | null> {
  try {
    // 1. Limpiar texto
    let cleanText = text
      .replace(/[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]/g, "") // quita caracteres raros
      .trim();

    // 2. Ignorar chunks vacíos
    if (!cleanText || cleanText.length < 5) {
      console.warn(
        `⚠️ Chunk ${i !== undefined ? "#" + (i + 1) : ""} omitido: texto vacío o demasiado corto`
      );
      return null;
    }

    // 3. Recortar texto si es muy largo
    const MAX_CHARS = 4000; // ajusta según el modelo
    if (cleanText.length > MAX_CHARS) {
      console.warn(
        `⚠️ Chunk ${i !== undefined ? "#" + (i + 1) : ""} recortado: longitud ${cleanText.length} → ${MAX_CHARS}`
      );
      cleanText = cleanText.slice(0, MAX_CHARS);
    }

    // 4. Llamar al servidor de embeddings
    const response = await fetch("http://10.0.0.17:1234/v1/embeddings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "text-embedding-granite-embedding-278m-multilingual",
        input: cleanText,
      }),
    });

    // 5. Manejo de error HTTP
    if (!response.ok) {
      const errorText = await response.text();
      console.error(
        `❌ Error HTTP ${response.status} al pedir embedding para chunk ${i !== undefined ? "#" + (i + 1) : ""} (longitud=${cleanText.length}):`,
        errorText
      );
      return null;
    }

    const data = (await response.json()) as EmbeddingResponse;

    if (!data?.data?.[0]?.embedding) {
      console.warn(
        `⚠️ No se recibió embedding válido para chunk ${i !== undefined ? "#" + (i + 1) : ""}: "${cleanText.slice(0, 80)}..."`
      );
      return null;
    }

    return data.data[0].embedding;
  } catch (err) {
    console.error(
      `❌ Error inesperado en getEmbedding para chunk ${i !== undefined ? "#" + (i + 1) : ""}: "${text.slice(0, 80)}..."`,
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
// 5. Cargar base en memoria con cache
// ----------------------
export async function loadKnowledge(): Promise<void> {
  const cachePath = path.join(__dirname, "BaseDeConocimiento.json");

  // Si ya existe cache, cargarlo
  if (fs.existsSync(cachePath)) {
    knowledgeBase = JSON.parse(fs.readFileSync(cachePath, "utf-8"));
    console.log(
      `✅ Base de conocimiento cargada desde cache con ${knowledgeBase.length} chunks`
    );
    return;
  }

  // Si no hay cache, procesar el documento
  const filePath = path.join(__dirname, "conocimiento.docx");
  const text = await loadWordFile(filePath);
  const chunks = chunkFAQs(text); // ✅ usamos FAQs en lugar de corte por palabras

   knowledgeBase = [];
  for (const [i, chunk] of chunks.entries()) {
    const embedding = await getEmbedding(chunk); // ✅ chunk es string
    if (!embedding) {
      console.warn(
        `⚠️ Chunk #${i + 1} omitido (sin embedding): "${chunk.slice(0, 80)}..."`
      );
      continue;
    }
    // ✅ construimos el objeto correctamente
    knowledgeBase.push({ text: chunk, embedding });
  }

  // Guardar en cache
  fs.writeFileSync(cachePath, JSON.stringify(knowledgeBase, null, 2));
  console.log(
    `✅ Base de conocimiento creada y guardada con ${knowledgeBase.length} chunks`
  );
}

// ----------------------
// 6. Recuperar contexto mejorado
// ----------------------
export async function retrieveContext(
  question: string,
  topN: number = 2,
  minWords: number = 5,
  minScore: number = 0.65
): Promise<string | null> {
  if (!knowledgeBase.length) return null;

  const qEmbedding = await getEmbedding(question);
  if (!qEmbedding) {
    console.warn("⚠️ No se pudo generar embedding para la pregunta.");
    return null;
  }

  // Calcular similitud con todos los chunks
  const ranked = knowledgeBase
    .map((chunk) => ({
      text: chunk.text,
      score: cosineSimilarity(qEmbedding, chunk.embedding),
    }))
    .sort((a, b) => b.score - a.score);

  // Mostrar top 5 resultados para depuración
  console.log(`\n🔎 Resultados de similitud para: "${question}"`);
  ranked.slice(0, 5).forEach((r, i) => {
    console.log(
      `   #${i + 1} → Score: ${r.score.toFixed(3)} | Texto: ${r.text.slice(0, 80)}...`
    );
  });

  // Filtrar chunks relevantes
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
// 7. Guardar preguntas sin respueta
// -----------------

const unansweredPath = path.join(__dirname, "new_questions.json");

interface UnansweredEntry {
  message: string;
  timestamp: number;
  context: string[];
  topScore: number;
}

// ------------------ Guardar mensaje no respondido con contexto parcial ------------------
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