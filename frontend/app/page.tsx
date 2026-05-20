"use client";
import Link from "next/link";
import {
  Zap, Shield, GitBranch, Search, ArrowRight, CheckCircle2
} from "lucide-react";

const features = [
  {
    icon: GitBranch,
    title: "Adaptive Query Routing",
    description:
      "Every query is classified as simple, complex, or hybrid and routed to the optimal retrieval strategy automatically.",
  },
  {
    icon: Search,
    title: "Hybrid Retrieval",
    description:
      "Combines dense vector search with BM25 sparse retrieval, fused via Reciprocal Rank Fusion for 30% better precision.",
  },
  {
    icon: Zap,
    title: "Cross-Encoder Reranking",
    description:
      "A 22M-param cross-encoder re-scores candidates with full query-document attention before passing chunks to the LLM.",
  },
  {
    icon: Shield,
    title: "Hallucination Detection",
    description:
      "Every answer is scored for faithfulness. Low-confidence answers are flagged so users know when to verify.",
  },
];

const comparisons = [
  { label: "Naive RAG", precision: "~55%", latency: "800ms", hallucination: "~25%" },
  { label: "SmartRAG", precision: "~82%", latency: "2.1s", hallucination: "<5%", highlight: true },
];

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-white">
      {/* Nav */}
      <nav className="border-b border-gray-100 px-6 py-4 flex items-center justify-between max-w-6xl mx-auto">
        <div className="flex items-center gap-2 font-semibold text-lg text-indigo-600">
          <Zap className="w-5 h-5" /> SmartRAG
        </div>
        <div className="flex items-center gap-4">
          <Link href="/dashboard" className="text-sm text-gray-600 hover:text-gray-900 transition-colors">
            Dashboard
          </Link>
          <Link
            href="/dashboard"
            className="text-sm bg-indigo-600 text-white px-4 py-2 rounded-lg hover:bg-indigo-700 transition-colors"
          >
            Get Started →
          </Link>
        </div>
      </nav>

      {/* Hero */}
      <section className="max-w-4xl mx-auto px-6 pt-24 pb-16 text-center">
        <div className="inline-flex items-center gap-2 text-xs font-medium bg-indigo-50 text-indigo-700 px-3 py-1 rounded-full mb-6">
          <span className="w-1.5 h-1.5 rounded-full bg-indigo-500 animate-pulse"></span>
          30/30 AI Projects — Day 2
        </div>
        <h1 className="text-5xl font-bold text-gray-900 leading-tight mb-6">
          RAG that actually works
          <br />
          <span className="text-indigo-600">in production.</span>
        </h1>
        <p className="text-xl text-gray-500 max-w-2xl mx-auto mb-10">
          SmartRAG combines hybrid retrieval, cross-encoder reranking, and hallucination detection
          — the things enterprise RAG systems spend months building, ready in one deploy.
        </p>
        <div className="flex items-center justify-center gap-4">
          <Link
            href="/dashboard"
            className="flex items-center gap-2 bg-indigo-600 text-white px-6 py-3 rounded-xl font-medium hover:bg-indigo-700 transition-colors"
          >
            Start for free <ArrowRight className="w-4 h-4" />
          </Link>
          <a
            href="http://localhost:8000/docs"
            target="_blank"
            className="flex items-center gap-2 border border-gray-200 text-gray-700 px-6 py-3 rounded-xl font-medium hover:border-gray-300 transition-colors"
          >
            API docs
          </a>
        </div>
      </section>

      {/* Stats */}
      <section className="bg-gray-50 border-y border-gray-100 py-10">
        <div className="max-w-4xl mx-auto px-6 grid grid-cols-3 gap-8 text-center">
          {[
            { val: "82%", label: "context precision (vs 55% naive)" },
            { val: "<5%", label: "hallucination rate" },
            { val: "30%", label: "precision gain from reranking" },
          ].map((s) => (
            <div key={s.label}>
              <p className="text-4xl font-bold text-indigo-600">{s.val}</p>
              <p className="text-sm text-gray-500 mt-1">{s.label}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Features */}
      <section className="max-w-6xl mx-auto px-6 py-20">
        <h2 className="text-3xl font-bold text-center text-gray-900 mb-4">
          Four innovations. One pipeline.
        </h2>
        <p className="text-center text-gray-500 mb-12 max-w-xl mx-auto">
          Each layer independently improves answer quality. Together they make SmartRAG
          the most accurate open-source RAG system available.
        </p>
        <div className="grid md:grid-cols-2 gap-6">
          {features.map((f) => (
            <div key={f.title} className="border border-gray-100 rounded-2xl p-6 hover:border-indigo-100 hover:shadow-sm transition-all">
              <div className="w-10 h-10 bg-indigo-50 rounded-lg flex items-center justify-center mb-4">
                <f.icon className="w-5 h-5 text-indigo-600" />
              </div>
              <h3 className="font-semibold text-gray-900 mb-2">{f.title}</h3>
              <p className="text-gray-500 text-sm leading-relaxed">{f.description}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Comparison */}
      <section className="bg-gray-50 border-y border-gray-100 py-20">
        <div className="max-w-3xl mx-auto px-6">
          <h2 className="text-3xl font-bold text-center text-gray-900 mb-12">
            SmartRAG vs. Naive RAG
          </h2>
          <div className="overflow-hidden rounded-2xl border border-gray-200">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-white border-b border-gray-100">
                  <th className="text-left py-4 px-6 text-gray-500 font-medium">System</th>
                  <th className="text-center py-4 px-4 text-gray-500 font-medium">Precision</th>
                  <th className="text-center py-4 px-4 text-gray-500 font-medium">Latency</th>
                  <th className="text-center py-4 px-4 text-gray-500 font-medium">Hallucination</th>
                </tr>
              </thead>
              <tbody>
                {comparisons.map((row) => (
                  <tr
                    key={row.label}
                    className={row.highlight ? "bg-indigo-50" : "bg-white"}
                  >
                    <td className={`py-4 px-6 font-medium ${row.highlight ? "text-indigo-700" : "text-gray-700"}`}>
                      {row.label}
                      {row.highlight && (
                        <span className="ml-2 text-xs bg-indigo-100 text-indigo-600 px-2 py-0.5 rounded-full">
                          this project
                        </span>
                      )}
                    </td>
                    <td className={`py-4 px-4 text-center ${row.highlight ? "text-indigo-700 font-semibold" : "text-gray-600"}`}>
                      {row.precision}
                    </td>
                    <td className={`py-4 px-4 text-center ${row.highlight ? "text-indigo-700 font-semibold" : "text-gray-600"}`}>
                      {row.latency}
                    </td>
                    <td className={`py-4 px-4 text-center ${row.highlight ? "text-indigo-700 font-semibold" : "text-gray-600"}`}>
                      {row.hallucination}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="max-w-2xl mx-auto px-6 py-24 text-center">
        <h2 className="text-3xl font-bold text-gray-900 mb-4">Ready to query your documents?</h2>
        <p className="text-gray-500 mb-8">Upload a PDF, ask a question, get a cited answer with a confidence score.</p>
        <Link
          href="/dashboard"
          className="inline-flex items-center gap-2 bg-indigo-600 text-white px-8 py-4 rounded-xl font-medium hover:bg-indigo-700 transition-colors text-lg"
        >
          Open dashboard <ArrowRight className="w-5 h-5" />
        </Link>
      </section>

      <footer className="border-t border-gray-100 py-8 text-center text-sm text-gray-400">
        SmartRAG — 30/30 AI Projects Day 2 · Built with FastAPI, LangChain, Next.js
      </footer>
    </div>
  );
}
