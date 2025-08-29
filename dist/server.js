// server.ts
import express from "express";
import cors from "cors";
import { v4 as uuidv4 } from "uuid";
import fetch from "node-fetch";
// -------------------
// Configuración express
// -------------------
const app = express();
app.use(cors({
    origin: "http://localhost:3000", // cámbialo a tu frontend real en prod
    methods: ["GET", "POST", "OPTIONS"],
    allowedHeaders: ["Content-Type", "Authorization"],
}));
app.options("*", cors());
app.use(express.json());
// -------------------
// Sesiones en memoria
// -------------------
const sessions = {};
// Crear nueva sesión con system prompt inicial
app.get("/new-session", (req, res) => {
    const sessionId = uuidv4();
    sessions[sessionId] = [
        { role: "system", content: "Eres un asistente útil y siempre respondes en español." }
    ];
    res.json({ sessionId });
});
// -------------------
// Chat endpoint
// -------------------
app.post("/chat", async (req, res) => {
    const { sessionId, message } = req.body;
    if (!sessionId || !message) {
        return res.status(400).json({ error: "sessionId y message son requeridos" });
    }
    if (!sessions[sessionId]) {
        sessions[sessionId] = [
            { role: "system", content: "Eres un asistente útil y siempre respondes en español." }
        ];
    }
    // Guardar mensaje del usuario
    sessions[sessionId].push({ role: "user", content: message });
    console.log("➡️ Petición recibida:", { sessionId, message });
    try {
        // Petición al LLM
        const llmRes = await fetch("http://127.0.0.1:1234/v1/chat/completions", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                model: "meta-llama-3-8b-instruct",
                messages: sessions[sessionId],
                temperature: 0.7,
                max_tokens: 512, // ⚡ evita -1
                stream: false // ⚡ corregido (minúscula)
            }),
        });
        const data = await llmRes.json();
        const botResponse = data?.choices?.[0]?.message?.content || "No entendí.";
        // Guardar respuesta en la sesión
        sessions[sessionId].push({ role: "assistant", content: botResponse });
        console.log("🤖 Respuesta del LLM:", botResponse);
        res.json({ textResponse: botResponse });
    }
    catch (error) {
        console.error("❌ Error al conectar con LLM:", error);
        res.status(500).json({ textResponse: "Error en el servidor." });
    }
});
// -------------------
// Server start
// -------------------
const PORT = 3001;
app.listen(PORT, () => console.log(`✅ Backend listo en http://localhost:${PORT}`));
//# sourceMappingURL=server.js.map