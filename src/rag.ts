import fs from "fs";
import path from "path";
import mammoth from "mammoth";
import fetch from "node-fetch";
import { fileURLToPath } from "url";

interface Chunk {
  text: string;
  embedding: number[];
}

// ----------------------
// ESM-friendly __dirname
// ----------------------
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Base de conocimiento en memoria
let knowledgeBase: Chunk[] = [];

// ----------------------
// 1. Leer archivo Word con Mammoth
// ----------------------
export async function loadWordFile(filePath: string): Promise<string> {
  const buffer = fs.readFileSync(filePath);
  const result = await mammoth.extractRawText({ buffer });
  return result.value;
}

// ----------------------
// 2. Dividir texto en chunks
// ----------------------
function chunkText(text: string, chunkSize = 500): string[] {
  const words = text.split(/\s+/);
  const chunks: string[] = [];
  for (let i = 0; i < words.length; i += chunkSize) {
    chunks.push(words.slice(i, i + chunkSize).join(" "));
  }
  return chunks;
}

// ----------------------
// 3. Generar embeddings
// ----------------------
interface EmbeddingResponse {
  data: { embedding: number[] }[];
}

export async function getEmbedding(text: string): Promise<number[]> {
  const res = await fetch("http://localhost:1234/v1/embeddings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "text-embedding-nomic-embed-text-v1.5",
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
function cosineSimilarity(a: number[], b: number[]): number {
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
// 5. Cargar base en memoria
// ----------------------
export async function loadKnowledge(): Promise<void> {
  const filePath = path.join(__dirname, "conocimiento.docx");
  const text = await loadWordFile(filePath);
  const chunks = chunkText(text);

  knowledgeBase = await Promise.all(
    chunks.map(async (chunk) => ({
      text: chunk,
      embedding: await getEmbedding(chunk),
    }))
  );

  console.log(`✅ Base de conocimiento cargada con ${knowledgeBase.length} chunks`);
}

// ----------------------
// 6. Recuperar contexto
// ----------------------
export async function retrieveContext(question: string): Promise<string> {
  if (!knowledgeBase.length) return "";

  const qEmbedding = await getEmbedding(question);

  const ranked = knowledgeBase
    .map((chunk) => ({
      text: chunk.text,
      score: cosineSimilarity(qEmbedding, chunk.embedding),
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, 3);

  return ranked.map((r) => r.text).join("\n\n");
}
