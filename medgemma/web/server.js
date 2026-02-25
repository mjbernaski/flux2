const express = require("express");
const { createProxyMiddleware } = require("http-proxy-middleware");
const path = require("path");

const app = express();
const PORT = 3000;
const API_TARGET = "http://127.0.0.1:5000";

// Proxy /api/* to the Python FastAPI backend
app.use(
  "/api",
  createProxyMiddleware({
    target: API_TARGET,
    changeOrigin: true,
  })
);

// Serve static frontend files
app.use(express.static(path.join(__dirname, "public")));

app.listen(PORT, "0.0.0.0", () => {
  console.log(`Web server running at http://0.0.0.0:${PORT}`);
  console.log(`Proxying /api/* to ${API_TARGET}`);
});
