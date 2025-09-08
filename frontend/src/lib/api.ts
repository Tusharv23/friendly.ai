import axios from 'axios';

// Allow override via environment variable VITE_API_BASE, fallback to localhost:5000
const api = axios.create({
  baseURL: (import.meta as any).env?.VITE_API_BASE || 'http://127.0.0.1:5000',
  timeout: 20000,
});

api.interceptors.response.use(r => r, err => {
  if (err.response) {
    console.error('API error', err.response.status, err.response.data);
  } else {
    console.error('API network error', err.message);
  }
  return Promise.reject(err);
});

export default api;
