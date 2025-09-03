import { getEmbedding, cosineSimilarity, knowledgeBase } from "./rag.js";  // tus chunks

// 🔑 Stopwords en español comunes
const STOPWORDS = new Set([
  "a", "acá", "ahí", "al", "algo", "algunas", "algunos", "allá", "allí",
  "ambos", "ante", "antes", "aquel", "aquella", "aquellas", "aquellos",
  "aquí", "arriba", "así", "atrás", "aun", "aunque", "bajo", "bien",
  "cabe", "cada", "casi", "cierto", "como", "con", "conmigo", "conseguir",
  "consigo", "contigo", "contra", "cual", "cuales", "cualquier", "cuando",
  "de", "debajo", "dejar", "del", "demás", "demasiado", "dentro", "desde",
  "donde", "dos", "el", "ella", "ellas", "ellos", "en", "entre", "era",
  "erais", "éramos", "eran", "eres", "es", "esa", "esas", "ese", "eso",
  "esos", "esta", "estaba", "estado", "estáis", "estamos", "estan",
  "estar", "este", "esto", "estos", "estoy", "fin", "fue", "fueron",
  "fui", "fuimos", "gueno", "ha", "hace", "haces", "hacéis", "hacemos",
  "hacen", "hacer", "hacia", "hago", "incluso", "jamás", "junto", "la",
  "largo", "las", "lo", "los", "mientras", "mio", "mis", "misma", "mismas",
  "mismo", "mismos", "modo", "mucho", "muy", "nos", "nosotros", "nuestra",
  "nuestras", "nuestro", "nuestros", "nunca", "otra", "otras", "otro",
  "otros", "para", "pero", "por", "porque", "primero", "puede", "pueden",
  "puedo", "pues", "que", "quien", "quienes", "quizas", "se", "segun",
  "ser", "si", "siendo", "sin", "sobre", "solamente", "solo", "somos",
  "soy", "su", "sus", "tal", "también", "tener", "tengo", "tiempo",
  "tiene", "tienen", "toda", "todas", "todo", "todos", "tras", "un",
  "una", "uno", "unos", "usted", "ustedes", "va", "vais", "vamos", "van",
  "varias", "varios", "verdad", "vosotras", "vosotros", "voy", "yo"
]);

const MIN_THRESHOLD = 0.65; // score mínimo para considerar un chunk
const MARGIN = 0.05;        // diferencia mínima entre top1 y top2
const MAX_CHUNKS = 3;

function extractKeywords(text: string): string[] {
  return text
    .toLowerCase()
    .split(/\W+/) // separar por no-letras
    .filter(w => w.length > 3 && !STOPWORDS.has(w));
}

export async function retrieveContext(question: string): Promise<string> {
  console.log("\n🔎 Pregunta:", question);

  const embedding = await getEmbedding(question);
  if (!embedding.length) return "";

  // Ranquear chunks
  const ranked = knowledge.map(k => ({
    text: k.text,
    score: cosineSimilarity(embedding, k.embedding),
  })).sort((a, b) => b.score - a.score);

  const best = ranked[0];
  if (!best || best.score < MIN_THRESHOLD) {
    console.log(`⚠️ Ningún chunk supera el umbral mínimo (${MIN_THRESHOLD})`);
    return "";
  }

  if (ranked[1] && (best.score - ranked[1].score < MARGIN)) {
    console.log(`⚠️ Pregunta ambigua → top1 (${best.score.toFixed(3)}) y top2 (${ranked[1].score.toFixed(3)}) muy cercanos (<${MARGIN})`);
    return "";
  }

  // 🔎 Verificación de palabras clave
  const keywords = extractKeywords(question);
  console.log("📌 Palabras clave detectadas:", keywords);

  const containsKeyword = keywords.some(kw =>
    best.text.toLowerCase().includes(kw)
  );

  if (!containsKeyword) {
    console.log("⚠️ Chunk con mejor score no contiene ninguna keyword de la pregunta.");
    console.log("   → Chunk:", best.text.slice(0, 80) + "...");
    return "";
  }

  // ✅ Chunks relevantes finales
  const relevant = ranked
    .filter(r => r.score >= MIN_THRESHOLD)
    .slice(0, MAX_CHUNKS);

  console.log(`📌 Contexto seleccionado (${relevant.length} chunks):`);
  relevant.forEach(r =>
    console.log(`   → Score: ${r.score.toFixed(3)} | Texto: ${r.text.slice(0, 80)}...`)
  );

  return relevant.map(r => r.text).join("\n\n");
}
