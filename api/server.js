const express = require("express");
const cors = require("cors");
const axios = require("axios");
require("dotenv").config();

const app = express();
app.use(cors());
app.use(express.json()); // Parse JSON requests

const PORT = process.env.PORT || 5000;

// Route to check scam
app.post("/check", async (req, res) => {
  try {
    const { text } = req.body;
    const flaskAPI = "http://127.0.0.1:5001/detect"; // Flask AI API
    const response = await axios.post(flaskAPI, { text });
    res.json({ result: response.data.result });
  } catch (error) {
    console.error("Error:", error);
    res.status(500).json({ error: "Server error" });
  }
});

// Start Server
app.listen(PORT, () => console.log(`🚀 Server running on port ${PORT}`));
