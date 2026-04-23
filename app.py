from __future__ import annotations

import json
import os
import sys
import traceback

from flask import Flask, jsonify, render_template_string, request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_missing = []

try:
    from runner import generate, run_multiple
except ImportError as e:
    _missing.append(f"runner: {e}")
    def generate(p, m): raise RuntimeError("runner.py not found")
    def run_multiple(p, m, n=3): raise RuntimeError("runner.py not found")

try:
    from scorer import (
        WEIGHTS, compute_final_score, score_consistency, score_edge_cases,
        score_format_match, score_instruction_following, score_label,
        score_semantic_accuracy,
    )
except ImportError as e:
    _missing.append(f"scorer: {e}")

try:
    from prompt_analyser import analyse_prompt
except ImportError as e:
    _missing.append(f"prompt_analyser: {e}")
    def analyse_prompt(p, m): raise RuntimeError("prompt_analyser.py not found")

test_sensitivity = None
try:
    from sensitivity import test_sensitivity as _ts
    test_sensitivity = _ts
except ImportError as e:
    _missing.append(f"sensitivity (optional): {e}")

if _missing:
    print("\nWarning: some modules could not be imported:")
    for m in _missing:
        print(f"  . {m}")

app = Flask(__name__)

# -----------------------------------------------------------------------------
# RESEARCH CONTENT — managed by project owner, not editable from the UI
# -----------------------------------------------------------------------------

RESEARCH_PAPERS = [
    {
        "id": "prompt_interpretation",
        "title": "How Models Interpret Prompts",
        "tag": "Prompt Engineering",
        "color": "#c49abd",
        "summary": "Investigates how LLMs parse prompts internally — covering tokenization effects, instruction vs. context interaction, formatting patterns, and ambiguity signals. Establishes the theoretical foundation for why prompt structure matters.",
        "key_findings": [
            "Tokenization splits affect internal activations and downstream output quality",
            "Instruction adherence weakens when directives are diluted by long or competing context",
            "Clear delimiters and explicit output schemas reduce interpretation variance",
            "Ambiguous prompts produce higher entropy in next-token prediction distributions",
        ],
        "hypotheses": [
            "H1: Higher ambiguity produces higher average token entropy",
            "H2: Instruction-first structure shows higher task adherence",
            "H3: Structured prompts with delimiters reduce output variance",
            "H4: Small prompt edits can produce large output shifts in sensitive tasks",
        ],
        "references": [
            "Brown et al. (2020) — Language Models are Few-Shot Learners. NeurIPS.",
            "Liu et al. (2023) — Lost in the Middle: How LMs Use Long Contexts.",
            "Holtzman et al. (2020) — The Curious Case of Neural Text Degeneration. ICLR.",
            "Vaswani et al. (2017) — Attention Is All You Need. NeurIPS.",
        ],
    },
    {
        "id": "hallucinations",
        "title": "Hallucinations & Uncertainty in LLMs",
        "tag": "Hallucination Detection",
        "color": "#d4887a",
        "summary": "Defines hallucination as overconfident plausible falsehoods, explores root causes (evaluation incentives, pretraining dynamics, data bias), and surveys current detection and mitigation strategies including semantic entropy probes.",
        "key_findings": [
            "LLMs evaluated purely on accuracy are incentivised to guess rather than admit uncertainty",
            "Pretraining on next-token prediction is poor at learning rare factual data (e.g. birthdays)",
            "High entropy can signal correct uncertainty OR confident nonsense — not a direct proxy",
            "Semantic Entropy Probes (SEPs) make hallucination detection 5-10x more efficient",
        ],
        "hypotheses": [
            "Penalising confident errors more than uncertainty reduces hallucination rate",
            "RAG (retrieval-augmented generation) is currently the most effective mitigation",
            "Chain-of-Thought reasoning can worsen hallucinations when the process is fabricated",
            "Ensemble UQ scoring outperforms single-method confidence estimation",
        ],
        "references": [
            "Kalai et al. (2025) — Why Language Models Hallucinate. arXiv:2509.04664.",
            "Correa (2025) — Hallucinations in LLMs: The Entropy Problem. Monostate.",
            "PrajnaAI (2025) — Uncertainty Quantification Matters in LLMs. Medium.",
            "UQLM Toolkit — github.com/cvs-health/uqlm",
        ],
    },
    {
        "id": "entropy_perplexity",
        "title": "Entropy & Perplexity as Alignment Signals",
        "tag": "Alignment Metrics",
        "color": "#7a9fc4",
        "summary": "Examines entropy and perplexity as measurable proxies for model uncertainty and alignment quality. Covers AI alignment principles, metric limitations, and the calibration gap between model confidence and actual correctness.",
        "key_findings": [
            "Entropy measures spread of next-token probability distribution — high means uncertain",
            "Perplexity quantifies model surprise; lower perplexity means more confident predictions",
            "RLHF sharpens token distributions but can reduce output diversity — the alignment tax",
            "Calibration gap: models can be confidently wrong — 80% probability does not mean 80% accuracy",
        ],
        "hypotheses": [
            "Semantic entropy is a better hallucination signal than token-level entropy alone",
            "Abstention mechanisms triggered by high uncertainty improve safety by up to 99%",
            "Sycophancy emerges when alignment training overrides factual confidence",
            "KL divergence from the base model is a practical alignment health metric",
        ],
        "references": [
            "Jonker & Gomstyn (n.d.) — What is AI Alignment? IBM Think.",
            "Correa (2025) — Hallucinations in LLMs: The Entropy Problem. Monostate.",
            "Walker (n.d.) — Perplexity in AI and NLP. klu.ai.",
        ],
    },
    {
        "id": "tokenization",
        "title": "Tokenization & Next-Token Prediction",
        "tag": "Model Internals",
        "color": "#7ab89a",
        "summary": "Explains the full tokenization pipeline, common algorithms (BPE, WordPiece, SentencePiece), subword effects on reasoning, and how softmax probability distributions drive generation and confidence scoring.",
        "key_findings": [
            "Tokenizers are separate modules from the LLM but strongly influence output quality",
            "Morphological misalignment forces the model to reconstruct meaning from broken subwords",
            "Multilingual unfairness: non-English languages consume disproportionately more tokens",
            "Alignment tuning (RLHF) warps the probability distribution — the alignment tax",
        ],
        "hypotheses": [
            "Token fragmentation correlates with reduced reasoning accuracy",
            "Token healing mitigates mid-token prompt boundary failures",
            "Branching Factor sharpens as generation progresses — model becomes more certain",
            "High semantic entropy at generation time is a red flag for hallucinated alignment",
        ],
        "references": [
            "Vaswani et al. (2017) — Attention Is All You Need. NeurIPS.",
            "Brown et al. (2020) — Language Models are Few-Shot Learners. NeurIPS.",
            "Walker (n.d.) — Perplexity in AI and NLP. klu.ai.",
        ],
    },
]

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PromptLab</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500&family=DM+Mono:wght@300;400&display=swap" rel="stylesheet">
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --cream:  #faf7f4;
  --warm:   #f3ede6;
  --warm2:  #e8ddd3;
  --warm3:  #ddd0c4;
  --ink:    #1e1a16;
  --ink2:   #3d3530;
  --ink3:   #6b5f56;
  --mist:   #a0908a;
  --blush:  #c49abd;
  --blush2: #ddbfd6;
  --blush3: #f5ecf3;
  --rose:   #d4887a;
  --sage:   #7ab89a;
  --sky:    #7a9fc4;
  --gold:   #c4a46a;
  --serif:  'Cormorant Garamond', Georgia, serif;
  --sans:   'DM Sans', system-ui, sans-serif;
  --mono:   'DM Mono', monospace;
  --r:      10px;
  --sh:     0 1px 16px rgba(30,26,22,0.07);
  --sh2:    0 6px 40px rgba(30,26,22,0.13);
}

html { background: var(--cream); color: var(--ink); font-family: var(--sans); font-size: 14px; }
body { min-height: 100vh; display: flex; flex-direction: column; }

/* grain */
body::before {
  content:''; position:fixed; inset:0; z-index:0; pointer-events:none;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='300' height='300'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.75' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='300' height='300' filter='url(%23n)' opacity='0.035'/%3E%3C/svg%3E");
  opacity:.5;
}

/* HEADER */
.hdr {
  position:sticky; top:0; z-index:200;
  background:rgba(250,247,244,.92); backdrop-filter:blur(14px);
  border-bottom:1px solid var(--warm2);
  height:58px; display:flex; align-items:center; padding:0 30px; gap:14px;
}
.wm { font-family:var(--serif); font-size:22px; font-weight:300; color:var(--ink); letter-spacing:.01em; }
.wm em { font-style:italic; color:var(--blush); }
.pill {
  font-family:var(--mono); font-size:9px; letter-spacing:.08em;
  padding:3px 9px; border-radius:20px;
  background:var(--blush3); color:var(--blush); border:1px solid var(--blush2);
}
.hdr-right { margin-left:auto; display:flex; align-items:center; gap:7px; font-size:11px; color:var(--mist); }
.dot { width:7px; height:7px; border-radius:50%; background:var(--sage); box-shadow:0 0 7px var(--sage); animation:breathe 3s ease-in-out infinite; }
@keyframes breathe{0%,100%{opacity:1}50%{opacity:.45}}

/* SHELL */
.shell { display:flex; flex:1; position:relative; z-index:1; }

/* SIDEBAR */
.nav {
  width:216px; flex-shrink:0;
  background:var(--warm); border-right:1px solid var(--warm2);
  padding:24px 0; display:flex; flex-direction:column; gap:1px;
}
.nlabel {
  font-family:var(--mono); font-size:9px; letter-spacing:.12em; text-transform:uppercase;
  color:var(--mist); padding:10px 20px 4px;
}
.nbtn {
  all:unset; cursor:pointer; display:flex; align-items:center; gap:9px;
  padding:10px 20px; font-size:13px; color:var(--ink3);
  transition:all .18s; width:100%; position:relative; border-left:2px solid transparent;
}
.nbtn:hover { background:var(--warm2); color:var(--ink); }
.nbtn.active { background:var(--blush3); color:var(--ink); border-left-color:var(--blush); }
.nico { font-size:14px; width:18px; text-align:center; }
.ndiv { height:1px; background:var(--warm2); margin:10px 20px; }

/* PAGE WRAP — content + side panel */
.pw { flex:1; display:grid; grid-template-columns:1fr 272px; overflow:hidden; }
.main { overflow-y:auto; padding:30px; display:flex; flex-direction:column; gap:22px; }
.side {
  border-left:1px solid var(--warm2); background:var(--warm);
  overflow-y:auto; padding:26px 20px; display:flex; flex-direction:column; gap:18px;
}

/* TABS */
.tab { display:none; flex-direction:column; gap:22px; }
.tab.active { display:flex; }
.sc { display:none; flex-direction:column; gap:16px; }
.sc.active { display:flex; }

/* CARDS */
.card { background:#fff; border:1px solid var(--warm2); border-radius:var(--r); box-shadow:var(--sh); overflow:hidden; }
.chead { padding:15px 20px; border-bottom:1px solid var(--warm2); display:flex; align-items:center; gap:10px; }
.ctitle { font-family:var(--serif); font-size:17px; font-weight:400; color:var(--ink); }
.ctag { font-family:var(--mono); font-size:9px; letter-spacing:.07em; padding:2px 8px; border-radius:20px; background:var(--blush3); color:var(--blush); border:1px solid var(--blush2); margin-left:auto; }
.cbody { padding:20px; }

/* SIDE PANEL */
.sey { font-family:var(--mono); font-size:9px; letter-spacing:.11em; text-transform:uppercase; color:var(--mist); margin-bottom:5px; }
.sti { font-family:var(--serif); font-size:19px; font-weight:300; font-style:italic; color:var(--ink); line-height:1.3; margin-bottom:10px; }
.sbody { font-size:12px; color:var(--ink3); line-height:1.8; }
.sbody p { margin-bottom:9px; }
.srule { height:1px; background:var(--warm2); }
.slist { display:flex; flex-direction:column; gap:9px; }
.sitem { display:flex; gap:8px; font-size:12px; color:var(--ink3); line-height:1.6; }
.sbullet { width:5px; height:5px; border-radius:50%; background:var(--blush); flex-shrink:0; margin-top:7px; }
.snote { background:var(--blush3); border:1px solid var(--blush2); border-radius:8px; padding:12px 14px; font-size:11px; color:var(--ink3); line-height:1.7; }
.snote strong { color:var(--ink); }

/* FORM */
.field { display:flex; flex-direction:column; gap:6px; }
.lbl { font-family:var(--mono); font-size:9px; color:var(--mist); letter-spacing:.08em; text-transform:uppercase; }
.lbl .opt { color:var(--warm3); }
input[type=text],input[type=number],select,textarea {
  background:var(--warm); border:1px solid var(--warm2); border-radius:7px;
  color:var(--ink); font-family:var(--mono); font-size:12px;
  padding:9px 12px; outline:none; width:100%; resize:vertical;
  transition:border-color .18s, box-shadow .18s;
}
input:focus,select:focus,textarea:focus { border-color:var(--blush); box-shadow:0 0 0 3px var(--blush3); }
.g2 { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
.g3 { display:grid; grid-template-columns:1fr 1fr 1fr; gap:14px; }
.tog-row { display:flex; align-items:center; gap:14px; flex-wrap:wrap; }
.tog { display:flex; align-items:center; gap:7px; cursor:pointer; font-size:12px; color:var(--ink3); user-select:none; }
.tog input[type=checkbox] { accent-color:var(--blush); width:13px; height:13px; }
.tog:hover { color:var(--ink); }

/* BUTTONS */
.btn { display:inline-flex; align-items:center; gap:7px; padding:9px 20px; border-radius:7px; font-family:var(--sans); font-size:13px; font-weight:500; cursor:pointer; border:none; transition:all .18s; }
.btn-p { background:var(--blush); color:#fff; }
.btn-p:hover { background:#b589ad; box-shadow:0 4px 18px rgba(196,154,189,.38); }
.btn-p:disabled { opacity:.35; cursor:not-allowed; box-shadow:none; }
.btn-g { background:transparent; color:var(--ink3); border:1px solid var(--warm2); }
.btn-g:hover { background:var(--warm); color:var(--ink); }

/* LOG */
.log { font-family:var(--mono); font-size:11px; color:var(--mist); background:var(--warm); border-radius:7px; padding:13px 15px; min-height:80px; max-height:150px; overflow-y:auto; white-space:pre-wrap; line-height:1.85; border:1px solid var(--warm2); }
.log .ok { color:var(--sage); }
.log .warn { color:var(--gold); }
.log .err { color:var(--rose); }
.log .info { color:var(--sky); }

/* SCORE RING */
.score-hero { display:flex; align-items:center; gap:26px; padding:22px; background:var(--warm); border-radius:9px; border:1px solid var(--warm2); }
.sring { position:relative; width:90px; height:90px; flex-shrink:0; }
.sring svg { transform:rotate(-90deg); }
.rtrack { fill:none; stroke:var(--warm2); stroke-width:5; }
.rfill { fill:none; stroke-width:5; stroke-linecap:round; stroke-dasharray:239; stroke-dashoffset:239; transition:stroke-dashoffset 1.2s cubic-bezier(.4,0,.2,1); }
.rnum { position:absolute; inset:0; display:flex; flex-direction:column; align-items:center; justify-content:center; }
.rnum .n { font-family:var(--serif); font-size:27px; font-weight:300; line-height:1; }
.rnum .d { font-family:var(--mono); font-size:9px; color:var(--mist); margin-top:2px; }
.smeta .sl { font-family:var(--serif); font-size:19px; font-weight:300; font-style:italic; }
.smeta .ss { font-size:11px; color:var(--mist); margin-top:4px; line-height:1.6; }

/* DIM BARS */
.dg { display:flex; flex-direction:column; gap:15px; }
.dr { display:grid; grid-template-columns:152px 1fr 38px; align-items:center; gap:13px; }
.dn { font-size:12px; color:var(--ink3); }
.dn .wt { font-family:var(--mono); font-size:9px; color:var(--mist); margin-left:3px; }
.dtrack { height:4px; background:var(--warm2); border-radius:2px; overflow:hidden; }
.dfill { height:100%; border-radius:2px; transition:width 1s cubic-bezier(.4,0,.2,1); }
.dpct { font-family:var(--mono); font-size:11px; text-align:right; }
.dnote { grid-column:2/-1; font-size:11px; color:var(--mist); margin-top:-7px; line-height:1.5; }

/* METRIC BOXES */
.mbox { background:var(--warm); border:1px solid var(--warm2); border-radius:8px; padding:15px; }
.mkey { font-family:var(--mono); font-size:9px; color:var(--mist); letter-spacing:.1em; text-transform:uppercase; margin-bottom:7px; }
.mval { font-family:var(--serif); font-size:29px; font-weight:300; line-height:1; color:var(--ink); }
.msub { font-size:11px; color:var(--mist); margin-top:5px; }

/* CHIPS */
.chip-row { display:flex; flex-wrap:wrap; gap:6px; }
.chip { display:inline-block; padding:3px 10px; border-radius:20px; font-size:11px; font-family:var(--mono); border:1px solid var(--warm2); color:var(--mist); }
.chip.miss { border-color:#d4a060; color:#8a6020; background:#fdf6ea; }
.chip.conf { border-color:#c89090; color:#904040; background:#fdf2f0; }
.chip.ok   { border-color:#90c0a0; color:#306040; background:#f0f9f4; }

/* OUTPUT PRE */
.opre { background:var(--warm); border:1px solid var(--warm2); border-radius:7px; padding:13px; font-family:var(--mono); font-size:11px; color:var(--ink3); max-height:200px; overflow-y:auto; white-space:pre-wrap; line-height:1.8; }

/* SUM ITEMS */
.sum-item { display:flex; align-items:flex-start; gap:8px; font-size:12px; color:var(--ink3); line-height:1.65; }

/* SENS */
.st { flex:1; height:5px; background:var(--warm2); border-radius:3px; overflow:hidden; }
.sf { height:100%; border-radius:3px; transition:width .9s; }

/* RESEARCH CARDS */
.rc {
  background:#fff; border:1px solid var(--warm2); border-radius:var(--r);
  box-shadow:var(--sh); overflow:hidden; cursor:pointer;
  transition:box-shadow .2s, transform .2s;
}
.rc:hover { box-shadow:var(--sh2); transform:translateY(-1px); }
.rc.open { box-shadow:var(--sh2); }
.rchead { padding:18px 20px; display:flex; align-items:flex-start; gap:13px; }
.rcdot { width:9px; height:9px; border-radius:50%; flex-shrink:0; margin-top:6px; }
.rctag { font-family:var(--mono); font-size:9px; letter-spacing:.07em; padding:2px 8px; border-radius:20px; display:inline-block; margin-bottom:5px; }
.rctitle { font-family:var(--serif); font-size:17px; font-weight:400; color:var(--ink); line-height:1.3; }
.rcchev { margin-left:auto; font-size:15px; color:var(--mist); transition:transform .25s; flex-shrink:0; padding-top:4px; }
.rc.open .rcchev { transform:rotate(180deg); }
.rcbody { display:none; padding:0 20px 20px; border-top:1px solid var(--warm2); }
.rc.open .rcbody { display:block; padding-top:16px; }
.rcsumm { font-size:12px; color:var(--ink3); line-height:1.8; margin-bottom:14px; }
.rcstitle { font-family:var(--mono); font-size:9px; letter-spacing:.1em; text-transform:uppercase; color:var(--mist); margin-bottom:7px; margin-top:13px; }
.rcfind { display:flex; gap:8px; font-size:12px; color:var(--ink3); line-height:1.6; margin-bottom:5px; }
.rcfdot { width:4px; height:4px; border-radius:50%; flex-shrink:0; margin-top:8px; }
.rcref { font-size:10px; color:var(--mist); font-family:var(--mono); margin-bottom:3px; padding-left:12px; }

/* SPINNER */
@keyframes spin { to { transform:rotate(360deg); } }
.spin { width:13px; height:13px; border:2px solid rgba(255,255,255,.3); border-top-color:#fff; border-radius:50%; animation:spin .65s linear infinite; display:none; }

/* SCROLLBAR */
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:var(--warm2); border-radius:3px; }

.slabel { font-family:var(--mono); font-size:9px; color:var(--mist); letter-spacing:.1em; text-transform:uppercase; margin-bottom:8px; }
</style>
</head>
<body>

<div class="hdr">
  <div class="wm">Prompt<em>Lab</em></div>
  <div class="hdr-right"><div class="dot"></div><span>Ollama connected</span></div>
</div>

<div class="shell">

  <nav class="nav">
    <div class="nlabel">Evaluate</div>
    <button class="nbtn active" onclick="switchTab('eval')" id="nav-eval"><span class="nico">◈</span> Run Evaluation</button>
    <button class="nbtn" onclick="switchTab('analyse')" id="nav-analyse"><span class="nico">◎</span> Pre-Analyse</button>
    <div class="ndiv"></div>
    <div class="nlabel">Tools</div>
    <button class="nbtn" onclick="switchTab('demo')" id="nav-demo"><span class="nico">▷</span> Quick Demo</button>
    <div class="ndiv"></div>
    <div class="nlabel">Research</div>
    <button class="nbtn" onclick="switchTab('research')" id="nav-research"><span class="nico">✦</span> Research Papers</button>
  </nav>

  <div class="pw">
    <main class="main">

      <!-- EVALUATE -->
      <div class="tab active" id="tab-eval">
        <div class="card">
          <div class="chead"><div class="ctitle">Evaluation Setup</div><div class="ctag">core tool</div></div>
          <div class="cbody">
            <div class="g3" style="margin-bottom:16px">
              <div class="field"><div class="lbl">Model</div><input type="text" id="model" value="llama3.1:8b" placeholder="e.g. mistral"></div>
              <div class="field"><div class="lbl">Consistency Runs</div><input type="number" id="runs" value="3" min="1" max="10"></div>
              <div style="display:flex;flex-direction:column;justify-content:flex-end;gap:10px">
                <label class="tog"><input type="checkbox" id="sensitivity"> Sensitivity test</label>
                <label class="tog"><input type="checkbox" id="jsonOut"> Save report.json</label>
              </div>
            </div>
            <div class="g2" style="margin-bottom:16px">
              <div class="field">
                <div class="lbl">Prompt</div>
                <textarea id="prompt" rows="7" placeholder="Paste your prompt here…"></textarea>
                <input type="file" accept=".txt" onchange="loadFile(this,'prompt')" style="font-size:11px;color:var(--mist);margin-top:5px">
              </div>
              <div class="field">
                <div class="lbl">Expected Output <span class="opt">(optional)</span></div>
                <textarea id="expected" rows="7" placeholder="Paste expected output to unlock format &amp; semantic scoring…"></textarea>
                <input type="file" accept=".txt" onchange="loadFile(this,'expected')" style="font-size:11px;color:var(--mist);margin-top:5px">
              </div>
            </div>
            <div style="display:flex;justify-content:flex-end;gap:10px">
              <button class="btn btn-g" onclick="clearAll()">Clear</button>
              <button class="btn btn-p" id="runBtn" onclick="runEval()"><div class="spin" id="spinner"></div>Run evaluation</button>
            </div>
          </div>
        </div>
        <div class="card">
          <div class="chead"><div class="ctitle">Console</div></div>
          <div class="cbody" style="padding:13px"><div class="log" id="log">Ready — fill in the fields above and click Run evaluation.</div></div>
        </div>
        <div id="results" style="display:none;flex-direction:column;gap:22px">
          <div class="card"><div class="chead"><div class="ctitle">Overall Score</div></div><div class="cbody"><div class="score-hero" id="scoreHero"></div></div></div>
          <div class="card"><div class="chead"><div class="ctitle">Dimension Breakdown</div></div><div class="cbody"><div class="dg" id="dimBreakdown"></div></div></div>
          <div class="card"><div class="chead"><div class="ctitle">Primary Output</div></div><div class="cbody" style="padding:13px"><div class="opre" id="primaryOutput"></div></div></div>
          <div class="card" id="sensCard" style="display:none"><div class="chead"><div class="ctitle">Sensitivity / Robustness</div></div><div class="cbody" id="sensContent"></div></div>
          <div class="card"><div class="chead"><div class="ctitle">Summary</div></div><div class="cbody" id="summaryContent"></div></div>
        </div>
      </div>

      <!-- PRE-ANALYSE -->
      <div class="tab" id="tab-analyse">
        <div class="card">
          <div class="chead"><div class="ctitle">Prompt Pre-Analysis</div><div class="ctag">static · no model call</div></div>
          <div class="cbody">
            <div class="field" style="margin-bottom:14px"><div class="lbl">Prompt Text</div><textarea id="aPrompt" rows="8" placeholder="Paste prompt to analyse before sending to any model…"></textarea></div>
            <div class="g2" style="margin-bottom:14px"><div class="field"><div class="lbl">Model Name <span class="opt">(for token counting)</span></div><input type="text" id="aModel" value="gpt-4"></div></div>
            <button class="btn btn-p" id="analyseBtn" onclick="runAnalyse()">Analyse prompt</button>
          </div>
        </div>
        <div class="card" id="analyseResult" style="display:none">
          <div class="chead"><div class="ctitle">Analysis Result</div></div>
          <div class="cbody">
            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:11px;margin-bottom:18px" id="analysisGrid"></div>
            <div class="slabel">Missing components</div>
            <div class="chip-row" id="missingChips" style="margin-bottom:14px"></div>
            <div class="slabel">Conflict flags</div>
            <div class="chip-row" id="conflictChips"></div>
          </div>
        </div>
      </div>

      <!-- DEMO -->
      <div class="tab" id="tab-demo">
        <div class="card">
          <div class="chead"><div class="ctitle">Quick Demo</div><div class="ctag">live generation</div></div>
          <div class="cbody">
            <div class="g2" style="margin-bottom:14px"><div class="field"><div class="lbl">Model</div><input type="text" id="dModel" value="llama3.1:8b"></div></div>
            <div class="field" style="margin-bottom:14px"><div class="lbl">Prompt</div><textarea id="dPrompt" rows="6" placeholder="Ask anything…"></textarea></div>
            <button class="btn btn-p" id="demoBtn" onclick="runDemo()"><div class="spin" id="demoSpin"></div>Send prompt</button>
          </div>
        </div>
        <div class="card" id="demoResult" style="display:none">
          <div class="chead"><div class="ctitle">Response</div><div id="demoMeta" style="margin-left:auto;font-family:var(--mono);font-size:10px;color:var(--mist)"></div></div>
          <div class="cbody" style="padding:13px"><div class="opre" id="demoOutput"></div></div>
        </div>
      </div>

      <!-- RESEARCH -->
      <div class="tab" id="tab-research">
        <div>
          <div style="font-family:var(--serif);font-size:30px;font-weight:300;font-style:italic;color:var(--ink);margin-bottom:6px">Research Library</div>
          <div style="font-size:12px;color:var(--mist);line-height:1.7">Four papers underpinning this project — click any card to expand findings, hypotheses, and references.</div>
        </div>
        <div id="researchCards" style="display:flex;flex-direction:column;gap:13px"></div>
      </div>

    </main>

    <!-- SIDE PANEL -->
    <aside class="side">

      <div class="sc active" id="side-eval">
        <div><div class="sey">About this tool</div><div class="sti">What does the evaluator measure?</div>
          <div class="sbody"><p>PromptLab scores how well a prompt aligns with a model's output across <strong>five weighted dimensions</strong>. It gives you numbers, not vibes.</p></div>
        </div>
        <div class="srule"></div>
        <div><div class="sey">The 5 dimensions</div>
          <div class="slist">
            <div class="sitem"><div class="sbullet"></div><div><strong>Instruction following (30%)</strong> — does the output actually do what the prompt asks? Scored by an LLM judge.</div></div>
            <div class="sitem"><div class="sbullet"></div><div><strong>Format match (25%)</strong> — does the structure match the expected output? Checks JSON, bullets, length, headers.</div></div>
            <div class="sitem"><div class="sbullet"></div><div><strong>Semantic accuracy (25%)</strong> — does the meaning match? Uses a cross-encoder for deep text comparison.</div></div>
            <div class="sitem"><div class="sbullet"></div><div><strong>Consistency (10%)</strong> — does the model give similar answers across multiple runs?</div></div>
            <div class="sitem"><div class="sbullet"></div><div><strong>Edge cases (10%)</strong> — does it handle unusual inputs gracefully?</div></div>
          </div>
        </div>
        <div class="srule"></div>
        <div class="snote"><strong>Tip:</strong> Paste an expected output to unlock Format Match and Semantic Accuracy. Without it, those two return a neutral 0.75 score.</div>
        <div class="srule"></div>
        <div><div class="sey">Sensitivity test</div>
          <div class="sbody"><p>Tick the sensitivity checkbox to also test <em>prompt robustness</em> — the tool generates 5 rephrasings of your prompt and measures how much the model's output changes. A fragile prompt gets very different answers from slightly different wording.</p></div>
        </div>
      </div>

      <div class="sc" id="side-analyse">
        <div><div class="sey">About this tool</div><div class="sti">Why pre-analyse a prompt?</div>
          <div class="sbody"><p>Before you send a prompt to a model, it's worth asking: is it well-formed? Pre-analysis catches structural problems <strong>without spending any tokens</strong>.</p></div>
        </div>
        <div class="srule"></div>
        <div><div class="sey">What it checks</div>
          <div class="slist">
            <div class="sitem"><div class="sbullet"></div><div><strong>Token count</strong> — BPE tokens consumed. Relevant for context window limits.</div></div>
            <div class="sitem"><div class="sbullet"></div><div><strong>Prompt type</strong> — structured, creative, factual, or instructional?</div></div>
            <div class="sitem"><div class="sbullet"></div><div><strong>Ambiguity score</strong> — ratio of vague words to total. High ambiguity means unpredictable output.</div></div>
            <div class="sitem"><div class="sbullet"></div><div><strong>Complexity score</strong> — composite of token count, sentence count, and instruction density.</div></div>
            <div class="sitem"><div class="sbullet"></div><div><strong>Missing components</strong> — flags if the prompt lacks a clear task, context, or format instruction.</div></div>
            <div class="sitem"><div class="sbullet"></div><div><strong>Conflict flags</strong> — detects contradictions like "be concise but comprehensive."</div></div>
          </div>
        </div>
        <div class="srule"></div>
        <div class="snote"><strong>Research link:</strong> Our prompt interpretation paper (H2, H3) shows that instruction-first, explicitly formatted prompts produce measurably higher task adherence.</div>
      </div>

      <div class="sc" id="side-demo">
        <div><div class="sey">About this tool</div><div class="sti">Quick model sanity check</div>
          <div class="sbody"><p>Send any prompt to Ollama and see the raw response. Useful for confirming the model is running, testing a phrasing quickly, or showing a live demo without running the full scorer.</p></div>
        </div>
        <div class="srule"></div>
        <div class="snote"><strong>Demo tip:</strong> Try this prompt for a compelling live demo — it hits hallucination territory and shows interesting model behaviour:<br><br><em style="color:var(--ink)">"In 3 bullet points, explain what perplexity measures in language models and why low perplexity does not guarantee factual accuracy."</em></div>
        <div class="srule"></div>
        <div><div class="sey">What you're seeing</div>
          <div class="sbody"><p>Response time and word count below the output give a quick sense of model confidence — fast, short responses often come from high-certainty (low-entropy) token distributions.</p></div>
        </div>
      </div>

      <div class="sc" id="side-research">
        <div><div class="sey">Research overview</div><div class="sti">Four papers, one pipeline</div>
          <div class="sbody"><p>Each paper directly informs a component of PromptLab. Together they form the theoretical foundation for measurable prompt engineering.</p></div>
        </div>
        <div class="srule"></div>
        <div class="slist">
          <div class="sitem"><div class="sbullet" style="background:#c49abd"></div><div><strong>Prompt Interpretation</strong> → motivates the sensitivity tester and structural pre-analysis</div></div>
          <div class="sitem"><div class="sbullet" style="background:#d4887a"></div><div><strong>Hallucinations &amp; Uncertainty</strong> → motivates the uncertainty scorer and LLM-as-judge dimension</div></div>
          <div class="sitem"><div class="sbullet" style="background:#7a9fc4"></div><div><strong>Entropy &amp; Perplexity</strong> → motivates alignment scoring and the perplexity-based uncertainty signal</div></div>
          <div class="sitem"><div class="sbullet" style="background:#7ab89a"></div><div><strong>Tokenization</strong> → motivates token count analysis and fragmentation-aware prompt design</div></div>
        </div>
        <div class="srule"></div>
        <div class="snote">Click any card to expand findings, hypotheses, and references.</div>
      </div>

    </aside>
  </div>
</div>

<script>
const PAPERS = """ + json.dumps(RESEARCH_PAPERS) + r""";

function init() { renderResearchCards(); }

function switchTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.nbtn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.sc').forEach(s => s.classList.remove('active'));
  document.getElementById('tab-'+name).classList.add('active');
  document.getElementById('nav-'+name).classList.add('active');
  document.getElementById('side-'+name).classList.add('active');
}

function esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}
function loadFile(input, id) {
  const f=input.files[0]; if(!f) return;
  const r=new FileReader(); r.onload=e=>{document.getElementById(id).value=e.target.result;}; r.readAsText(f);
}
function clearAll() {
  ['prompt','expected'].forEach(id=>document.getElementById(id).value='');
  document.getElementById('results').style.display='none';
  document.getElementById('log').textContent='Cleared.';
}
function scolor(p) {
  if(p>=85) return '#7ab89a';
  if(p>=65) return '#c4a46a';
  if(p>=40) return '#d4887a';
  return '#c47878';
}
function log(msg,type='') {
  const el=document.getElementById('log');
  const ts=new Date().toLocaleTimeString('en',{hour12:false});
  const cls=type?` class="${esc(type)}"` :'';
  el.innerHTML+=`<span${cls}>[${ts}] ${esc(msg)}\n</span>`;
  el.scrollTop=el.scrollHeight;
}

async function runEval() {
  const prompt=document.getElementById('prompt').value.trim();
  const expected=document.getElementById('expected').value.trim();
  const model=document.getElementById('model').value.trim()||'llama3.1:8b';
  const runs=parseInt(document.getElementById('runs').value)||3;
  const sensitivity=document.getElementById('sensitivity').checked;
  const jsonOut=document.getElementById('jsonOut').checked;
  if(!prompt){alert('Please enter a prompt.');return;}
  const btn=document.getElementById('runBtn'), sp=document.getElementById('spinner');
  btn.disabled=true; sp.style.display='inline-block';
  document.getElementById('results').style.display='none';
  document.getElementById('log').innerHTML='';
  log(`Model: ${model} | Runs: ${runs} | Sensitivity: ${sensitivity}`,'info');
  log('Step 1/4 — Running prompt against model…');
  try {
    const res=await fetch('/api/evaluate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt,expected,model,runs,sensitivity,json_out:jsonOut})});
    if(!res.ok){const err=await res.json().catch(()=>({error:res.statusText}));log(`Error: ${err.error}`,'err');return;}
    const data=await res.json();
    log('Step 2/4 — Scoring complete','ok');
    if(sensitivity) log('Step 3/4 — Sensitivity test complete','ok');
    log('Step 4/4 — Rendering report…','ok');
    renderResults(data);
  } catch(e){log(`Connection failed: ${e.message}`,'err');log('Make sure Ollama is running (ollama serve).','warn');}
  finally{btn.disabled=false; sp.style.display='none';}
}

function renderResults(data) {
  const {final_score,dimensions,outputs,sensitivity:sens}=data;
  const pct=Math.round(final_score), col=scolor(pct);
  const label=pct>=85?'Well Aligned':pct>=65?'Mostly Aligned':pct>=40?'Partially Aligned':'Misaligned';
  const offset=239-(pct/100)*239;
  document.getElementById('scoreHero').innerHTML=`
    <div class="sring">
      <svg width="90" height="90" viewBox="0 0 90 90">
        <circle class="rtrack" cx="45" cy="45" r="38"/>
        <circle class="rfill" cx="45" cy="45" r="38" stroke="${col}" style="stroke-dashoffset:${offset}"/>
      </svg>
      <div class="rnum"><div class="n" style="color:${col}">${pct}</div><div class="d">/ 100</div></div>
    </div>
    <div class="smeta"><div class="sl" style="color:${col}">${esc(label)}</div><div class="ss">Weighted composite across 5 evaluation dimensions</div></div>`;
  const dims=[['instruction_following','Instruction following',30],['format_match','Format match',25],['semantic_accuracy','Semantic accuracy',25],['consistency','Consistency',10],['edge_cases','Edge cases',10]];
  let dh='';
  for(const [k,l,w] of dims){const s=dimensions[k];if(!s)continue;const p=Math.round(s.score*100),c=scolor(p);dh+=`<div class="dr"><div class="dn">${esc(l)} <span class="wt">(${w}%)</span></div><div class="dtrack"><div class="dfill" style="width:${p}%;background:${c}"></div></div><div class="dpct" style="color:${c}">${p}%</div><div class="dnote">${esc(s.note||'')}</div></div>`;}
  document.getElementById('dimBreakdown').innerHTML=dh;
  document.getElementById('primaryOutput').textContent=outputs?.[0]||'(no output)';
  if(sens){
    const sp=Math.round(sens.sensitivity_score*100);
    const [rl,rc]=sp<30?['Robust','#7ab89a']:sp<60?['Moderate','#c4a46a']:['Fragile','#c47878'];
    const stb=sens.most_stable_variation, uns=sens.least_stable_variation;
    document.getElementById('sensCard').style.display='';
    document.getElementById('sensContent').innerHTML=`<div style="display:flex;align-items:center;gap:11px;margin-bottom:12px"><span style="font-family:var(--mono);font-size:9px;padding:2px 9px;border-radius:20px;background:${rc}22;color:${rc};border:1px solid ${rc}44">${esc(rl)}</span><span style="font-size:12px;color:var(--ink3)">Sensitivity: ${sp}%</span></div><div style="display:flex;align-items:center;gap:11px;margin-bottom:7px"><div class="st"><div class="sf" style="width:${sp}%;background:${scolor(100-sp)}"></div></div><span style="font-family:var(--mono);font-size:10px;color:var(--mist)">${sp}%</span></div><p style="font-size:11px;color:var(--mist);margin-bottom:14px">Higher = more fragile to rewording.</p>${stb?`<div class="slabel">Most stable variation</div><div class="opre" style="max-height:70px;margin-bottom:11px">${esc(stb.variation_text)}</div>`:''}${uns?`<div class="slabel">Least stable variation</div><div class="opre" style="max-height:70px">${esc(uns.variation_text)}</div>`:''}`;
  }
  const weak=dims.filter(([k])=>{const s=dimensions[k];return s&&s.score<0.65;});
  let sh=!weak.length?`<div class="sum-item"><span style="color:#7ab89a">✓</span><span style="color:#406050">No major issues detected. The prompt is well-aligned with this model.</span></div>`:weak.map(([,l])=>`<div class="sum-item" style="margin-bottom:7px"><span style="color:#c4a46a">⚠</span><span><strong>${esc(l)}</strong> is below the 65% threshold — review the notes above.</span></div>`).join('');
  if(sens&&sens.sensitivity_score>=0.6) sh+=`<div class="sum-item" style="margin-top:7px"><span style="color:#c47878">⚠</span><span>This prompt is <strong>fragile</strong> — small rewording changes produce very different outputs.</span></div>`;
  document.getElementById('summaryContent').innerHTML=sh;
  const r=document.getElementById('results'); r.style.display='flex'; r.style.flexDirection='column';
}

async function runAnalyse() {
  const prompt=document.getElementById('aPrompt').value.trim();
  const model=document.getElementById('aModel').value.trim()||'gpt-4';
  if(!prompt){alert('Enter a prompt.');return;}
  const btn=document.getElementById('analyseBtn'); btn.disabled=true; btn.textContent='Analysing…';
  try {
    const res=await fetch('/api/analyse',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt,model})});
    const data=await res.json(); if(!res.ok){alert(data.error||'Error');return;}
    document.getElementById('analyseResult').style.display='';
    document.getElementById('analysisGrid').innerHTML=`<div class="mbox"><div class="mkey">Token count</div><div class="mval">${esc(String(data.token_count))}</div><div class="msub">BPE tokens</div></div><div class="mbox"><div class="mkey">Prompt type</div><div class="mval" style="font-size:17px;text-transform:capitalize">${esc(data.prompt_type)}</div><div class="msub">Detected intent</div></div><div class="mbox"><div class="mkey">Ambiguity</div><div class="mval">${Math.round(data.ambiguity_score*100)}%</div><div class="msub">Vague-word ratio</div></div><div class="mbox"><div class="mkey">Complexity</div><div class="mval">${Math.round(data.complexity_score*100)}%</div><div class="msub">Composite score</div></div><div class="mbox"><div class="mkey">Est. output</div><div class="mval" style="font-size:17px;text-transform:capitalize">${esc(data.estimated_output_length)}</div><div class="msub">Length estimate</div></div>`;
    const miss=data.missing_components||[]; document.getElementById('missingChips').innerHTML=miss.length?miss.map(m=>`<span class="chip miss">${esc(m)}</span>`).join(''):'<span class="chip ok">none detected</span>';
    const conf=data.conflict_flags||[]; document.getElementById('conflictChips').innerHTML=conf.length?conf.map(c=>`<span class="chip conf">${esc(c)}</span>`).join(''):'<span class="chip ok">none detected</span>';
  } catch(e){alert(e.message);}
  finally{btn.disabled=false;btn.textContent='Analyse prompt';}
}

async function runDemo() {
  const prompt=document.getElementById('dPrompt').value.trim();
  const model=document.getElementById('dModel').value.trim()||'llama3.1:8b';
  if(!prompt){alert('Enter a prompt.');return;}
  const btn=document.getElementById('demoBtn'),sp=document.getElementById('demoSpin');
  btn.disabled=true; sp.style.display='inline-block';
  document.getElementById('demoResult').style.display='none';
  const t0=Date.now();
  try {
    const res=await fetch('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt,model})});
    const data=await res.json(); if(!res.ok){alert(data.error||'Error');return;}
    const el=((Date.now()-t0)/1000).toFixed(1);
    document.getElementById('demoOutput').textContent=data.output;
    document.getElementById('demoMeta').textContent=`${el}s · ${data.output.split(/\s+/).length} words`;
    document.getElementById('demoResult').style.display='';
  } catch(e){alert(e.message);}
  finally{btn.disabled=false;sp.style.display='none';}
}

function renderResearchCards() {
  document.getElementById('researchCards').innerHTML=PAPERS.map(p=>`
    <div class="rc" id="rc-${esc(p.id)}" onclick="toggleRC('${esc(p.id)}')">
      <div class="rchead">
        <div class="rcdot" style="background:${esc(p.color)}"></div>
        <div style="flex:1">
          <span class="rctag" style="background:${esc(p.color)}22;color:${esc(p.color)};border:1px solid ${esc(p.color)}55">${esc(p.tag)}</span>
          <div class="rctitle">${esc(p.title)}</div>
        </div>
        <div class="rcchev">⌄</div>
      </div>
      <div class="rcbody">
        <div class="rcsumm">${esc(p.summary)}</div>
        <div class="rcstitle">Key findings</div>
        ${p.key_findings.map(f=>`<div class="rcfind"><div class="rcfdot" style="background:${esc(p.color)}"></div><div>${esc(f)}</div></div>`).join('')}
        <div class="rcstitle">Hypotheses</div>
        ${p.hypotheses.map(h=>`<div class="rcfind"><div class="rcfdot" style="background:${esc(p.color)}88"></div><div>${esc(h)}</div></div>`).join('')}
        <div class="rcstitle">References</div>
        ${p.references.map(r=>`<div class="rcref">— ${esc(r)}</div>`).join('')}
      </div>
    </div>`).join('');
}

function toggleRC(id) {
  const c=document.getElementById('rc-'+id), was=c.classList.contains('open');
  document.querySelectorAll('.rc').forEach(x=>x.classList.remove('open'));
  if(!was) c.classList.add('open');
}

init();
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/api/generate", methods=["POST"])
def api_generate():
    data   = request.get_json(silent=True) or {}
    prompt = str(data.get("prompt", "")).strip()
    model  = str(data.get("model", "llama3.1:8b"))
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    try:
        return jsonify({"output": generate(prompt, model)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/evaluate", methods=["POST"])
def api_evaluate():
    data          = request.get_json(silent=True) or {}
    prompt_text   = str(data.get("prompt", "")).strip()
    expected_text = str(data.get("expected", "")).strip() or None
    model         = str(data.get("model", "llama3.1:8b"))
    runs          = max(1, min(int(data.get("runs", 3)), 10))
    run_sens      = bool(data.get("sensitivity", False))
    json_out      = bool(data.get("json_out", False))
    if not prompt_text:
        return jsonify({"error": "No prompt provided"}), 400
    try:
        outputs = run_multiple(prompt_text, model, n=runs)
        if not outputs or not any(outputs):
            return jsonify({"error": f"Model '{model}' returned empty output. Is Ollama running?"}), 500
        primary     = outputs[0]
        dimensions  = {
            "instruction_following": score_instruction_following(prompt_text, primary, model),
            "format_match":          score_format_match(expected_text, primary),
            "semantic_accuracy":     score_semantic_accuracy(expected_text, primary),
            "consistency":           score_consistency(outputs),
            "edge_cases":            score_edge_cases([]),
        }
        final  = compute_final_score(dimensions)
        result = {"final_score": final, "dimensions": dimensions, "outputs": outputs}
        if run_sens and test_sensitivity is not None:
            try:
                result["sensitivity"] = test_sensitivity(prompt_text, model, num_variations=5)
            except Exception as e:
                print(f"Sensitivity test failed: {e}")
        if json_out:
            with open("report.json", "w") as f:
                json.dump(result, f, indent=2)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/analyse", methods=["POST"])
def api_analyse():
    data   = request.get_json(silent=True) or {}
    prompt = str(data.get("prompt", "")).strip()
    model  = str(data.get("model", "gpt-4"))
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    try:
        return jsonify(analyse_prompt(prompt, model))
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("\n  PromptLab — http://localhost:5000\n")
    app.run(debug=True, port=5000)