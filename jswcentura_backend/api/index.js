// backend/index.js
const express = require("express");
const cors = require("cors");
const app = express();

app.use(cors());


app.post("/check-text", (req, res) => {
  res.json({ success: true });
});

app.listen(3000, () => console.log("Server running"));
