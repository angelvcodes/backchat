import express from "express";
import cors from "cors";

const app = express();
app.use(cors());
app.use(express.json());

// Almacenamos las sesiones en memoria con timestamp de última actividad
interface Session {
  messages: any[];
  lastActive: number;
}
const sessions: Record<string, Session> = {};

// Tiempo de expiración de sesión en milisegundos (ej: 30 minutos)
const SESSION_EXPIRATION = 30 * 60 * 1000;
//const SESSION_EXPIRATION = 10 * 1000; // 10 segundos para pruebas
// Función para limpiar sesiones expiradas
const cleanExpiredSessions = () => {
  const now = Date.now();
  for (const sessionId in sessions) {
  const session = sessions[sessionId];
  if (session && (Date.now() - session.lastActive > SESSION_EXPIRATION)) {
    console.log(`🗑️ Sesión ${sessionId} eliminada por inactividad`);
    delete sessions[sessionId];
  }
}
};

// Ejecutar limpieza cada 5 minutos
setInterval(cleanExpiredSessions, 5 * 60 * 1000);
//setInterval(cleanExpiredSessions, 5000);

// Recibir mensaje
app.post("/chat", (req, res) => {
  const { sessionId, message } = req.body;

  if (!sessionId) {
    return res.status(400).json({ error: "Falta sessionId" });
  }

  // Crear sesión si no existe
  if (!sessions[sessionId]) {
    sessions[sessionId] = { messages: [], lastActive: Date.now() };
  }

  // Actualizar última actividad
  sessions[sessionId].lastActive = Date.now();

  // Guardar mensaje de usuario
  sessions[sessionId].messages.push({ role: "user", content: message });

  // Respuesta de IA
  const respuesta = `Recibí tu mensaje: "${message}"`;

  // Guardar respuesta de asistente
  sessions[sessionId].messages.push({ role: "assistant", content: respuesta });

  // DEBUG: imprimir conversación
  console.log(`\n📌 Conversación [${sessionId}] (actualizada)`);
  sessions[sessionId].messages.forEach((msg, i) => {
    console.log(`${i + 1}. ${msg.role.toUpperCase()}: ${msg.content}`);
  });

  res.json({ textResponse: respuesta });
});

// Endpoint opcional para historial
app.get("/history/:sessionId", (req, res) => {
  const { sessionId } = req.params;
  if (!sessions[sessionId]) {
    return res.status(404).json({ error: "Sesión no encontrada" });
  }
  res.json(sessions[sessionId].messages);
});

app.listen(3001, () => console.log("🚀 Backend en http://localhost:3001"));
