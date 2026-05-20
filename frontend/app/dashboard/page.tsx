"use client";
import { useState, useEffect, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import axios from "axios";
import ReactMarkdown from "react-markdown";
import {
  Upload, FileText, Trash2, Search, Zap, Shield,
  CheckCircle2, AlertTriangle, Clock, ChevronDown, ChevronUp,
  LogOut, BarChart3
} from "lucide-react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface Document {
  id: string; filename: string; status: string;
  chunk_count: number; size_bytes: number; created_at: string;
}

interface Source {
  chunk_id: string; document_id: string; document_name: string;
  content: string; relevance_score: number; page_number?: number;
}

interface QueryResult {
  query_id: string; query: string; answer: string;
  query_type: string; sources: Source[];
  hallucination_score: number; is_flagged: boolean;
  confidence: string; latency_ms: number;
}

function useApi(apiKey: string) {
  return axios.create({
    baseURL: API_BASE,
    headers: { Authorization: `Bearer ${apiKey}` },
  });
}

function ConfidenceBadge({ confidence, flagged }: { confidence: string; flagged: boolean }) {
  const map = {
    high: "bg-green-50 text-green-700 border-green-200",
    medium: "bg-yellow-50 text-yellow-700 border-yellow-200",
    low: "bg-red-50 text-red-700 border-red-200",
  };
  return (
    <span className={`inline-flex items-center gap-1 text-xs px-2 py-1 rounded-full border font-medium ${map[confidence as keyof typeof map] || map.medium}`}>
      {flagged ? <AlertTriangle className="w-3 h-3" /> : <CheckCircle2 className="w-3 h-3" />}
      {confidence} confidence
    </span>
  );
}

function QueryTypeBadge({ type }: { type: string }) {
  const map: Record<string, string> = {
    simple: "bg-blue-50 text-blue-600",
    complex: "bg-purple-50 text-purple-600",
    hybrid: "bg-orange-50 text-orange-600",
  };
  return (
    <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${map[type] || "bg-gray-100 text-gray-600"}`}>
      {type} query
    </span>
  );
}

export default function Dashboard() {
  const [apiKey, setApiKey] = useState("");
  const [inputKey, setInputKey] = useState("");
  const [documents, setDocuments] = useState<Document[]>([]);
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<QueryResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [expandedSources, setExpandedSources] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"query" | "docs" | "register">("register");
  const [registerForm, setRegisterForm] = useState({ name: "", email: "" });
  const [newApiKey, setNewApiKey] = useState("");

  const api = useApi(apiKey);

  const fetchDocuments = useCallback(async () => {
    if (!apiKey) return;
    try {
      const { data } = await api.get("/documents/");
      setDocuments(data);
    } catch {}
  }, [apiKey]);

  useEffect(() => { fetchDocuments(); }, [fetchDocuments]);

  const onDrop = useCallback(async (files: File[]) => {
    if (!apiKey || !files.length) return;
    setUploading(true);
    for (const file of files) {
      const formData = new FormData();
      formData.append("file", file);
      try {
        await api.post("/documents/upload", formData, {
          headers: { "Content-Type": "multipart/form-data" },
        });
      } catch (e: any) {
        alert(`Upload failed: ${e.response?.data?.detail || e.message}`);
      }
    }
    setUploading(false);
    setTimeout(fetchDocuments, 1500);
  }, [apiKey, fetchDocuments]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "application/pdf": [], "text/plain": [],
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [] },
  });

  const handleQuery = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || !apiKey) return;
    setLoading(true);
    try {
      const { data } = await api.post("/query/", { query, top_k: 5 });
      setResults((prev) => [data, ...prev]);
      setQuery("");
    } catch (e: any) {
      alert(`Query failed: ${e.response?.data?.detail || e.message}`);
    }
    setLoading(false);
  };

  const handleRegister = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const { data } = await axios.post(`${API_BASE}/auth/register`, registerForm);
      setNewApiKey(data.api_key);
    } catch (e: any) {
      alert(`Registration failed: ${e.response?.data?.detail || e.message}`);
    }
  };

  const handleDelete = async (docId: string) => {
    try {
      await api.delete(`/documents/${docId}`);
      fetchDocuments();
    } catch {}
  };

  const formatBytes = (b: number) =>
    b > 1024 * 1024 ? `${(b / 1024 / 1024).toFixed(1)}MB` : `${(b / 1024).toFixed(0)}KB`;

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      {/* Header */}
      <header className="bg-white border-b border-gray-100 px-6 py-4">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-2 font-semibold text-indigo-600">
            <Zap className="w-5 h-5" /> SmartRAG
          </div>
          {apiKey && (
            <button
              onClick={() => setApiKey("")}
              className="flex items-center gap-1 text-sm text-gray-500 hover:text-gray-700"
            >
              <LogOut className="w-4 h-4" /> Sign out
            </button>
          )}
        </div>
      </header>

      <div className="max-w-6xl mx-auto w-full flex-1 px-6 py-8 flex gap-6">
        {/* Sidebar Tabs */}
        <aside className="w-48 shrink-0 space-y-1">
          {[
            { id: "register", label: "Get API Key", icon: Shield },
            { id: "docs", label: "Documents", icon: FileText },
            { id: "query", label: "Ask Questions", icon: Search },
          ].map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setActiveTab(id as any)}
              className={`w-full flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                activeTab === id
                  ? "bg-indigo-50 text-indigo-700"
                  : "text-gray-600 hover:bg-gray-100"
              }`}
            >
              <Icon className="w-4 h-4" /> {label}
            </button>
          ))}
        </aside>

        {/* Main Content */}
        <main className="flex-1 space-y-6">

          {/* ── Register Tab ── */}
          {activeTab === "register" && (
            <div className="bg-white rounded-2xl border border-gray-100 p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-1">Get your API key</h2>
              <p className="text-sm text-gray-500 mb-6">Register once to get a free API key, then paste it below to start uploading documents.</p>

              {!newApiKey ? (
                <form onSubmit={handleRegister} className="space-y-4 max-w-sm">
                  <input
                    className="w-full border border-gray-200 rounded-lg px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-300"
                    placeholder="Your name"
                    value={registerForm.name}
                    onChange={(e) => setRegisterForm({ ...registerForm, name: e.target.value })}
                    required
                  />
                  <input
                    className="w-full border border-gray-200 rounded-lg px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-300"
                    placeholder="your@email.com"
                    type="email"
                    value={registerForm.email}
                    onChange={(e) => setRegisterForm({ ...registerForm, email: e.target.value })}
                    required
                  />
                  <button
                    type="submit"
                    className="bg-indigo-600 text-white px-6 py-2 rounded-lg text-sm font-medium hover:bg-indigo-700 transition-colors"
                  >
                    Register & get key
                  </button>
                </form>
              ) : (
                <div className="space-y-4 max-w-sm">
                  <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                    <p className="text-xs font-medium text-green-700 mb-2">Your API key (save this!):</p>
                    <code className="text-xs text-green-800 break-all">{newApiKey}</code>
                  </div>
                  <button
                    onClick={() => { setApiKey(newApiKey); setActiveTab("docs"); }}
                    className="bg-indigo-600 text-white px-6 py-2 rounded-lg text-sm font-medium hover:bg-indigo-700 transition-colors"
                  >
                    Continue to dashboard →
                  </button>
                </div>
              )}

              <div className="mt-8 pt-6 border-t border-gray-100">
                <p className="text-xs text-gray-400 mb-2">Already have a key?</p>
                <div className="flex gap-2 max-w-sm">
                  <input
                    className="flex-1 border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-300"
                    placeholder="srag_..."
                    value={inputKey}
                    onChange={(e) => setInputKey(e.target.value)}
                  />
                  <button
                    onClick={() => { setApiKey(inputKey); setActiveTab("docs"); }}
                    disabled={!inputKey}
                    className="bg-gray-800 text-white px-4 py-2 rounded-lg text-sm font-medium hover:bg-gray-900 disabled:opacity-40 transition-colors"
                  >
                    Load
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* ── Documents Tab ── */}
          {activeTab === "docs" && (
            <div className="space-y-4">
              {/* Upload zone */}
              <div
                {...getRootProps()}
                className={`border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer transition-colors ${
                  isDragActive ? "border-indigo-400 bg-indigo-50" : "border-gray-200 bg-white hover:border-gray-300"
                }`}
              >
                <input {...getInputProps()} />
                <Upload className="w-8 h-8 text-gray-400 mx-auto mb-3" />
                {uploading ? (
                  <p className="text-sm text-indigo-600 font-medium">Uploading & indexing…</p>
                ) : (
                  <>
                    <p className="text-sm font-medium text-gray-700">Drop files here or click to upload</p>
                    <p className="text-xs text-gray-400 mt-1">PDF, DOCX, TXT · max 50MB</p>
                  </>
                )}
              </div>

              {/* Document list */}
              {documents.length > 0 && (
                <div className="bg-white rounded-2xl border border-gray-100 overflow-hidden">
                  <div className="px-5 py-3 border-b border-gray-50 flex items-center justify-between">
                    <span className="text-sm font-medium text-gray-700">
                      {documents.length} document{documents.length !== 1 ? "s" : ""}
                    </span>
                    <button onClick={fetchDocuments} className="text-xs text-indigo-500 hover:underline">
                      Refresh
                    </button>
                  </div>
                  <ul className="divide-y divide-gray-50">
                    {documents.map((doc) => (
                      <li key={doc.id} className="px-5 py-3 flex items-center gap-3">
                        <FileText className="w-4 h-4 text-gray-400 shrink-0" />
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-medium text-gray-800 truncate">{doc.filename}</p>
                          <p className="text-xs text-gray-400">
                            {formatBytes(doc.size_bytes)} ·{" "}
                            {doc.chunk_count > 0 ? `${doc.chunk_count} chunks` : "processing…"}
                          </p>
                        </div>
                        <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                          doc.status === "ready" ? "bg-green-50 text-green-600" :
                          doc.status === "failed" ? "bg-red-50 text-red-600" :
                          "bg-yellow-50 text-yellow-600"
                        }`}>
                          {doc.status}
                        </span>
                        <button
                          onClick={() => handleDelete(doc.id)}
                          className="p-1 text-gray-300 hover:text-red-400 transition-colors"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {documents.length === 0 && !uploading && (
                <div className="bg-white rounded-2xl border border-gray-100 p-12 text-center">
                  <FileText className="w-8 h-8 text-gray-200 mx-auto mb-3" />
                  <p className="text-sm text-gray-400">No documents yet. Upload your first file above.</p>
                </div>
              )}
            </div>
          )}

          {/* ── Query Tab ── */}
          {activeTab === "query" && (
            <div className="space-y-4">
              {/* Query form */}
              <form onSubmit={handleQuery} className="bg-white rounded-2xl border border-gray-100 p-4">
                <div className="flex gap-3">
                  <input
                    className="flex-1 border border-gray-200 rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-300"
                    placeholder="Ask a question about your documents…"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    disabled={loading}
                  />
                  <button
                    type="submit"
                    disabled={loading || !query.trim()}
                    className="bg-indigo-600 text-white px-6 py-3 rounded-xl text-sm font-medium hover:bg-indigo-700 disabled:opacity-50 transition-colors flex items-center gap-2"
                  >
                    {loading ? (
                      <span className="flex items-center gap-2">
                        <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                        Thinking…
                      </span>
                    ) : (
                      <><Search className="w-4 h-4" /> Ask</>
                    )}
                  </button>
                </div>
              </form>

              {/* Results */}
              {results.map((result) => (
                <div key={result.query_id} className="bg-white rounded-2xl border border-gray-100 p-5 space-y-4">
                  {/* Meta row */}
                  <div className="flex flex-wrap items-center gap-2">
                    <QueryTypeBadge type={result.query_type} />
                    <ConfidenceBadge confidence={result.confidence} flagged={result.is_flagged} />
                    <span className="flex items-center gap-1 text-xs text-gray-400">
                      <Clock className="w-3 h-3" /> {result.latency_ms}ms
                    </span>
                    <span className="flex items-center gap-1 text-xs text-gray-400">
                      H-score: {(result.hallucination_score * 100).toFixed(0)}%
                    </span>
                  </div>

                  {/* Question */}
                  <p className="text-sm font-medium text-gray-500">Q: {result.query}</p>

                  {/* Hallucination warning */}
                  {result.is_flagged && (
                    <div className="flex items-start gap-2 bg-amber-50 border border-amber-100 rounded-xl p-3 text-xs text-amber-700">
                      <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
                      <span>
                        <strong>Verify this answer.</strong> The confidence score is low — some claims may not be fully supported by the retrieved documents.
                      </span>
                    </div>
                  )}

                  {/* Answer */}
                  <div className="prose prose-sm max-w-none text-gray-800">
                    <ReactMarkdown>{result.answer}</ReactMarkdown>
                  </div>

                  {/* Sources */}
                  {result.sources.length > 0 && (
                    <div>
                      <button
                        className="flex items-center gap-1 text-xs text-indigo-500 hover:underline"
                        onClick={() =>
                          setExpandedSources(expandedSources === result.query_id ? null : result.query_id)
                        }
                      >
                        {expandedSources === result.query_id ? (
                          <ChevronUp className="w-3 h-3" />
                        ) : (
                          <ChevronDown className="w-3 h-3" />
                        )}
                        {result.sources.length} source chunk{result.sources.length !== 1 ? "s" : ""}
                      </button>

                      {expandedSources === result.query_id && (
                        <div className="mt-3 space-y-2">
                          {result.sources.map((s, i) => (
                            <div key={s.chunk_id} className="border border-gray-100 rounded-xl p-3 text-xs">
                              <div className="flex items-center gap-2 mb-2">
                                <span className="font-medium text-gray-600">
                                  [{i + 1}] {s.document_name}
                                </span>
                                {s.page_number && (
                                  <span className="text-gray-400">p.{s.page_number}</span>
                                )}
                                <span className="ml-auto text-indigo-500 font-medium">
                                  {(s.relevance_score * 100).toFixed(0)}% relevance
                                </span>
                              </div>
                              <p className="text-gray-500 leading-relaxed line-clamp-3">{s.content}</p>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))}

              {results.length === 0 && (
                <div className="bg-white rounded-2xl border border-gray-100 p-12 text-center">
                  <Search className="w-8 h-8 text-gray-200 mx-auto mb-3" />
                  <p className="text-sm text-gray-400">
                    Ask a question to see SmartRAG in action.
                    <br />Make sure you have documents uploaded first.
                  </p>
                </div>
              )}
            </div>
          )}

        </main>
      </div>
    </div>
  );
}
