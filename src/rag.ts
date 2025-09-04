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
  data: { embedding: number[] }[];
}

export async function getEmbedding(text: string): Promise<number[]> {
  const res = await fetch("http://10.0.0.17:1234/v1/embeddings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "text-embedding-granite-embedding-278m-multilingual",
      input: text,
    }),
  });

  const data = (await res.json()) as EmbeddingResponse;
  if (!data?.data?.[0]?.embedding) {
    throw new Error("No se recibió embedding válido del modelo.");
  }
  return data.data[0].embedding;
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
  const cachePath = path.join(__dirname, "knowledgeBase.json");

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
  for (let i = 0; i < chunks.length; i++) {
    const chunk = chunks[i]!;
    try {
      const embedding = await getEmbedding(chunk);
      knowledgeBase.push({ text: chunk, embedding, id: i });
    } catch (err) {
      console.error("❌ Error generando embedding para chunk:", err);
    }

    // ⚠️ Evita saturar el modelo (espera 200ms entre requests)
    await new Promise((res) => setTimeout(res, 200));
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
      `   #${i + 1} → Score: ${r.score.toFixed(3)} | Texto: ${r.text.slice(
        0,
        80
      )}...`
    );
  });

  // Filtrar chunks relevantes por umbral y longitud mínima
  const relevant = ranked
    .filter((r) => r.score >= minScore && r.text.split(" ").length >= minWords)
    .slice(0, topN);

  // Si no hay resultados suficientemente relevantes
  if (relevant.length === 0) {
    console.log("⚠️ No se encontró contexto relevante para esta pregunta.");
    return null; // Devuelve null para indicar ausencia de contexto
  }

  console.log(
    `📌 Contexto seleccionado (${relevant.length} chunks sobre threshold ${minScore} y minWords ${minWords}):`
  );
  relevant.forEach((r) =>
    console.log(
      `   → Score: ${r.score.toFixed(3)} | Texto: ${r.text.slice(0, 80)}...`
    )
  );

  // Devolver texto concatenado de los chunks relevantes
  return relevant.map((r) => r.text).join("\n\n");
}