// src/api.js
import axios from "axios";

// Use /api in prod (App Engine dispatch), localhost in dev if you prefer
const API_BASE_URL = process.env.REACT_APP_API_BASE || "/api";

const api = axios.create({
    baseURL: API_BASE_URL,
    timeout: 30000,
});

export { API_BASE_URL };
export default api;
