"""
SmartRAG Demo Seeder
Creates a demo tenant and uploads sample documents so you can test queries immediately.
Usage: python scripts/seed_demo.py
"""
import asyncio
import httpx

API_BASE = "http://localhost:8000"

SAMPLE_TEXTS = [
    {
        "filename": "refund_policy.txt",
        "content": """SmartCorp Refund Policy (v3.2)

Customers may request a full refund within 30 days of purchase for any reason.
After 30 days and within 90 days, a 50% partial refund is available for unused software licences.
Physical products must be returned in original packaging within 14 days. Shipping costs are non-refundable.

To initiate a refund, email support@smartcorp.com with your order ID and reason.
Refunds are processed within 5-7 business days to the original payment method.

Exceptions: Custom enterprise contracts, annual plans discounted over 40%, and one-time setup fees are non-refundable.
""",
    },
    {
        "filename": "q4_strategy.txt",
        "content": """Q4 2025 Strategic Priorities — SmartCorp Internal

Revenue targets: $12M ARR by December 31. Current ARR: $9.3M. Gap: $2.7M.

Three growth pillars:
1. Enterprise expansion: Close 3 Fortune 500 deals in pipeline (Acme, WidgetCo, TechBigCo).
   Average deal size $600K. Estimated close: 70% probability combined.
2. APAC market entry: Launch Singapore office Q4. First 6 months goal: 5 mid-market accounts.
   Pricing strategy: 20% below US list price to gain share. Review in Q2 2026.
3. Product-led growth: Launch freemium tier October 1. Target 10,000 signups, 5% conversion to paid.

Key risks: Sales headcount (2 AEs open, recruiting in progress), APAC regulatory compliance (legal review 90% done).
""",
    },
    {
        "filename": "technical_faq.txt",
        "content": """SmartCorp API — Developer FAQ

Q: What authentication method does the API use?
A: The API uses Bearer token authentication. Include your API key in every request:
   Authorization: Bearer your-api-key-here

Q: What are the rate limits?
A: Free tier: 100 requests/hour. Pro tier: 1,000 requests/hour. Enterprise: unlimited with SLA.

Q: Which file formats are supported for document upload?
A: PDF, DOCX, TXT, MD, CSV. Maximum file size: 50MB. Maximum documents per account: 500.

Q: How long does document indexing take?
A: Small documents (<10 pages): 30-60 seconds. Large documents (100+ pages): 2-5 minutes.
   Poll the GET /documents endpoint to check status.

Q: Can I filter queries to specific documents?
A: Yes. Pass filter_document_ids: ["doc_id_1", "doc_id_2"] in the query request body.
""",
    },
]


async def main():
    async with httpx.AsyncClient(base_url=API_BASE, timeout=30.0) as client:
        print("Creating demo tenant...")
        r = await client.post("/auth/register", json={"name": "Demo User", "email": "demo@smartrag.test"})
        if r.status_code not in (200, 201):
            print(f"Registration failed (may already exist): {r.text}")
            return

        api_key = r.json()["api_key"]
        headers = {"Authorization": f"Bearer {api_key}"}
        print(f"API Key: {api_key}")
        print(f"\nUploading {len(SAMPLE_TEXTS)} sample documents...")

        for sample in SAMPLE_TEXTS:
            files = {"file": (sample["filename"], sample["content"].encode(), "text/plain")}
            r = await client.post("/documents/upload", headers=headers, files=files)
            print(f"  ✓ {sample['filename']} → {r.status_code}")

        print("\nWaiting 10s for indexing...")
        await asyncio.sleep(10)

        print("\nTest queries:")
        test_queries = [
            "What is the refund policy after 30 days?",
            "What are the Q4 revenue targets and the gap to close?",
            "How do I authenticate with the API?",
        ]

        for q in test_queries:
            r = await client.post("/query/", headers=headers, json={"query": q, "top_k": 3})
            if r.status_code == 200:
                data = r.json()
                print(f"\nQ: {q}")
                print(f"A: {data['answer'][:200]}...")
                print(f"   Type: {data['query_type']} | Confidence: {data['confidence']} | {data['latency_ms']}ms")
            else:
                print(f"Query failed: {r.text}")

        print("\n✓ Demo seeding complete!")
        print(f"  Frontend: http://localhost:3000")
        print(f"  API docs: http://localhost:8000/docs")
        print(f"  Your API key: {api_key}")


if __name__ == "__main__":
    asyncio.run(main())
