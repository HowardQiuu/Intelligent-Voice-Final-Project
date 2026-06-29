export const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";
export const MAX_BROWSER_UPLOAD_MB = Number(import.meta.env.VITE_MAX_BROWSER_UPLOAD_MB || 120);
export const UPLOAD_CHUNK_MB = Number(import.meta.env.VITE_UPLOAD_CHUNK_MB || 4);

export function apiUrl(path) {
  if (path?.startsWith("/")) return `${API_BASE}${path}`;
  return path;
}

export async function fetchDemoCases() {
  const response = await fetch(`${API_BASE}/api/demo-cases`);
  if (!response.ok) throw new Error("demo cases unavailable");
  return response.json();
}

export async function processDemo(caseId) {
  const response = await fetch(`${API_BASE}/api/process-demo/${caseId}`, { method: "POST" });
  if (!response.ok) throw new Error("demo process failed");
  return response.json();
}

export async function uploadAudioFile(file, onProgress) {
  const session = await createUploadSession(file);
  const chunkSize = session.chunk_size_bytes || UPLOAD_CHUNK_MB * 1024 * 1024;
  const totalChunks = session.total_chunks || Math.ceil(file.size / chunkSize);

  for (let index = 0; index < totalChunks; index += 1) {
    const start = index * chunkSize;
    const end = Math.min(file.size, start + chunkSize);
    const form = new FormData();
    form.append("file", file.slice(start, end), `${file.name}.part${index}`);
    const response = await fetch(`${API_BASE}/api/upload-session/${session.upload_id}/chunk?index=${index}`, {
      method: "POST",
      body: form,
    });
    if (!response.ok) throw new Error("chunk upload failed");
    onProgress?.({
      phase: "uploading",
      percent: Math.round(((index + 1) / totalChunks) * 100),
      uploadedBytes: end,
      totalBytes: file.size,
    });
  }

  onProgress?.({ phase: "processing", percent: 100, uploadedBytes: file.size, totalBytes: file.size });
  const complete = await fetch(`${API_BASE}/api/upload-session/${session.upload_id}/complete`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ filename: file.name, total_chunks: totalChunks }),
  });
  if (!complete.ok) throw new Error("upload complete failed");
  return complete.json();
}

export async function processLocalFile(path) {
  const response = await fetch(`${API_BASE}/api/process-local-file`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  });
  if (!response.ok) throw new Error("local file process failed");
  return response.json();
}

async function createUploadSession(file) {
  const response = await fetch(`${API_BASE}/api/upload-session`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ filename: file.name, size_bytes: file.size }),
  });
  if (!response.ok) throw new Error("upload session failed");
  return response.json();
}
