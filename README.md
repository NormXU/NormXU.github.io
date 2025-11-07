---
title: "About Me"
permalink: "/about/"
layout: page
---

<style>
:root{
  --bg:#ffffff;
  --card:#f8fafc;
  --text:#0f172a;
  --muted:#475569;
  --brand:#2563eb;
  --brand-2:#7c3aed;
  --ring:rgba(37,99,235,.25);
  --shadow:0 10px 25px rgba(2,6,23,0.08);
  --radius:16px;
}
@media (prefers-color-scheme: dark){
  :root{
    --bg:#0b1020;
    --card:#0f172a;
    --text:#e5e7eb;
    --muted:#94a3b8;
    --brand:#60a5fa;
    --brand-2:#a78bfa;
    --ring:rgba(96,165,250,.35);
    --shadow:0 10px 25px rgba(0,0,0,.35);
  }
}
.about-page{
  font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial;
  color:var(--text);
}
.about-page .wrap{
  max-width:900px;
  margin:0 auto;
  padding:1rem 1.25rem 3rem;
}
.hero{
  text-align:center;
  background:linear-gradient(135deg,color-mix(in oklab,var(--brand) 16%,transparent),color-mix(in oklab,var(--brand-2) 12%,transparent));
  border-radius:var(--radius);
  padding:1.5rem 1.25rem;
  box-shadow:var(--shadow);
}
.hero h1{
  margin:0;
  font-size:clamp(1.8rem,2vw + 1rem,2.4rem);
}
.hero p{
  margin:.5rem 0 0;
  color:var(--muted);
  font-size:1.05rem;
}
.hero .actions{
  margin-top:1rem;
  display:flex;
  gap:.6rem;
  justify-content:center;
  flex-wrap:wrap;
}
.hero .btn{
  display:inline-flex;
  align-items:center;
  gap:.5rem;
  padding:.6rem .9rem;
  border-radius:999px;
  border:1px solid color-mix(in oklab,var(--brand) 45%,transparent);
  text-decoration:none;
  color:var(--text);
  background:color-mix(in oklab,var(--brand) 12%,var(--card));
  transition:transform .12s ease,box-shadow .12s ease,background .12s ease;
}
.hero .btn:hover{
  transform:translateY(-1px);
  box-shadow:var(--shadow);
  background:color-mix(in oklab,var(--brand) 18%,var(--card));
}
.hero .btn.secondary{
  border-color:color-mix(in oklab,var(--brand-2) 45%,transparent);
  background:color-mix(in oklab,var(--brand-2) 10%,var(--card));
}
.hero .btn.secondary:hover{
  background:color-mix(in oklab,var(--brand-2) 16%,var(--card));
}
.section{margin-top:2rem;}
.section h2{
  font-size:1.1rem;
  letter-spacing:.03em;
  text-transform:uppercase;
  color:var(--muted);
  margin-bottom:.75rem;
}
.tags{display:flex;gap:.5rem;flex-wrap:wrap;justify-content:center;}
.tag{
  font-size:.9rem;
  padding:.35rem .6rem;
  border-radius:999px;
  background:var(--card);
  border:1px solid color-mix(in oklab,var(--brand) 25%,transparent);
}
.lead{font-size:1.02rem;color:var(--muted);text-align:center;}
.card{
  background:var(--card);
  border:1px solid color-mix(in oklab,var(--brand) 15%,transparent);
  border-radius:var(--radius);
  padding:1rem;
  box-shadow:var(--shadow);
  transition:transform .12s ease,border-color .12s ease,box-shadow .12s ease;
}
.card:hover{
  transform:translateY(-2px);
  border-color:color-mix(in oklab,var(--brand) 35%,transparent);
  box-shadow:0 16px 35px rgba(2,6,23,0.12);
}
.timeline{display:grid;gap:.75rem;}
.timeline-item{display:grid;grid-template-columns:14px 1fr;gap:.75rem;align-items:flex-start;}
.dot{
  width:14px;height:14px;border-radius:999px;margin-top:.35rem;
  background:radial-gradient(circle at 30% 30%,var(--brand),var(--brand-2));
  box-shadow:0 0 0 4px color-mix(in oklab,var(--ring) 50%,transparent);
}
.grid{display:grid;gap:1rem;}
@media (min-width:720px){.grid{grid-template-columns:1fr 1fr;}}
.work-title{font-weight:600;margin:0 0 .25rem;font-size:1.02rem;}
.work-meta{color:var(--muted);font-size:.95rem;margin-bottom:.5rem;}
.link-row{display:flex;gap:.5rem;flex-wrap:wrap;}
.link{
  font-size:.92rem;text-decoration:none;padding:.35rem .6rem;border-radius:8px;
  background:color-mix(in oklab,var(--brand) 10%,var(--bg));
  border:1px solid color-mix(in oklab,var(--brand) 35%,transparent);
  color:var(--text);
}
.link:hover{background:color-mix(in oklab,var(--brand) 16%,var(--bg));}
</style>

<div class="about-page">
  <div class="wrap">

    <section class="hero">
      <h1>Nuo Xu</h1>
      <p>Recent News: Building a multimodal Omni-Model @ Meituan LongCat Team.</p>
      <div class="actions">
        <a class="btn secondary" href="https://github.com/NormXU" target="_blank" rel="noopener">𐄷 GitHub</a>
      </div>
    </section>

    <section class="section">
      <h2>Research Interests</h2>
      <div class="tags">
        <span class="tag">Large Language Models</span>
        <span class="tag">Multimodal Learning</span>
        <span class="tag">Video Generation</span>
        <span class="tag">Data/Training Systems</span>
      </div>
    </section>

    <section class="section">
      <p class="lead">This site catalogs ideas, notes, and project updates. I hope some of it proves useful or sparks collaboration.</p>
    </section>

    <section class="section">
      <h2>Experience</h2>
      <div class="card">
        <div class="timeline">
          <div class="timeline-item"><div class="dot"></div><div><strong>Meituan • LongCat Team</strong> — Multimodal / Omni-Model</div></div>
          <div class="timeline-item"><div class="dot"></div><div><strong>Moonshot AI</strong> — Video Generation</div></div>
          <div class="timeline-item"><div class="dot"></div><div><strong>DataGrand</strong> — OCR</div></div>
        </div>
      </div>
    </section>

    <section class="section">
      <h2>Recent Works</h2>
      <div class="grid">
        <article class="card">
          <h3 class="work-title">LongCat-Flash-Omni Technical Report</h3>
          <div class="work-meta">Team, Meituan LongCat, <em>arXiv:2511.00279</em> (2025)</div>
          <div class="link-row">
            <a class="link" href="https://github.com/meituan-longcat/LongCat-Flash-Omni" target="_blank" rel="noopener">GitHub</a>
            <a class="link" href="https://arxiv.org/abs/2511.00279" target="_blank" rel="noopener">Paper</a>
          </div>
        </article>

        <article class="card">
          <h3 class="work-title">Kimi-VL Technical Report</h3>
          <div class="work-meta">Team, Kimi, <em>arXiv:2504.07491</em> (2025)</div>
          <div class="link-row">
            <a class="link" href="https://github.com/MoonshotAI/Kimi-VL" target="_blank" rel="noopener">GitHub</a>
            <a class="link" href="https://arxiv.org/abs/2504.07491" target="_blank" rel="noopener">Paper</a>
          </div>
        </article>

        <article class="card">
          <h3 class="work-title">PARAGRAPH2GRAPH: A GNN-based Framework for Layout Paragraph Analysis</h3>
          <div class="work-meta">Wei, Shu & <strong>Nuo Xu</strong>, <em>arXiv:2304.11810</em> (2023)</div>
          <div class="link-row">
            <a class="link" href="https://github.com/NormXU/Layout2Graph" target="_blank" rel="noopener">GitHub</a>
            <a class="link" href="https://arxiv.org/abs/2304.11810" target="_blank" rel="noopener">Paper</a>
          </div>
        </article>
      </div>
    </section>

  </div>
</div>
