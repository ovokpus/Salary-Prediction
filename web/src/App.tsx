import { useEffect, useMemo, useState } from "react";

type Posting = { companyId: string; jobType: string; degree: string; major: string; industry: string; yearsExperience: number; milesFromMetropolis: number };
type Options = Record<"companyId" | "jobType" | "degree" | "major" | "industry", string[]>;
type Insights = Record<string, Record<string, number>>;
type Metrics = Record<string, number | string>;

const post = (body: unknown) => fetch("/api/predict", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) }).then((r) => r.json());
const LABELS: Record<string, string> = { jobType: "Job type", companyId: "Company", industry: "Industry", degree: "Degree", major: "Major" };
const nice = (s: string) => LABELS[s] ?? (["CEO", "CFO", "CTO"].includes(s) ? s : s.replace(/_/g, " ").toLowerCase().replace(/\b\w/g, (c) => c.toUpperCase()));

function BarChart({ data, highlight }: { data: Record<string, number>; highlight?: string }) {
  const entries = Object.entries(data);
  const max = Math.max(...entries.map(([, v]) => v));
  return (
    <div className="chart">
      {entries.map(([k, v]) => (
        <div key={k} className={`hrow ${k === highlight ? "hl" : ""}`}>
          <span className="lbl">{nice(k)}</span>
          <div className="track"><div style={{ width: `${(v / max) * 100}%` }} /></div>
          <span className="val">${v.toFixed(0)}k</span>
        </div>
      ))}
    </div>
  );
}

export default function App() {
  const [opts, setOpts] = useState<Options | null>(null);
  const [ins, setIns] = useState<Insights | null>(null);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [p, setP] = useState<Posting>({ companyId: "COMP37", jobType: "SENIOR", degree: "MASTERS", major: "ENGINEERING", industry: "FINANCE", yearsExperience: 10, milesFromMetropolis: 20 });
  const [salary, setSalary] = useState<number | null>(null);
  const [curve, setCurve] = useState<number[]>([]);
  const [chart, setChart] = useState("jobType");

  useEffect(() => {
    fetch("/api/options").then((r) => r.json()).then(setOpts);
    fetch("/api/insights").then((r) => r.json()).then(setIns);
    fetch("/api/model").then((r) => r.json()).then(setMetrics);
  }, []);

  useEffect(() => {
    const t = setTimeout(async () => {
      const years = Array.from({ length: 25 }, (_, i) => i);
      const d = await post({ postings: [p, ...years.map((y) => ({ ...p, yearsExperience: y }))] });
      setSalary(d.predictions[0].salary_k);
      setCurve(d.predictions.slice(1).map((x: { salary_k: number }) => x.salary_k));
    }, 120);
    return () => clearTimeout(t);
  }, [p]);

  const sel = (k: keyof Options) => (
    <label className="field"><span>{nice(k)}</span>
      <select value={p[k]} onChange={(e) => setP({ ...p, [k]: e.target.value })}>
        {opts?.[k].map((o) => <option key={o} value={o}>{k === "companyId" ? o : nice(o)}</option>)}
      </select>
    </label>
  );

  const path = useMemo(() => {
    if (!curve.length) return "";
    const lo = Math.min(...curve) - 5, hi = Math.max(...curve) + 5;
    return curve.map((v, i) => `${i ? "L" : "M"}${(i / 24) * 300},${120 - ((v - lo) / (hi - lo)) * 120}`).join(" ");
  }, [curve]);

  return (
    <div className="page">
      <header>
        <h1>💼 Salary Predictor</h1>
        <p>Estimate a job posting's salary from its role, education, industry, experience and distance from the city. Trained on 1M postings.</p>
      </header>
      <div className="grid">
        <section className="card">
          <h2>Job posting</h2>
          <div className="two">{sel("jobType")}{sel("industry")}{sel("degree")}{sel("major")}{sel("companyId")}</div>
          <label className="field"><span>Years of experience <b>{p.yearsExperience}</b></span>
            <input type="range" min={0} max={24} value={p.yearsExperience} onChange={(e) => setP({ ...p, yearsExperience: +e.target.value })} /></label>
          <label className="field"><span>Miles from metropolis <b>{p.milesFromMetropolis}</b></span>
            <input type="range" min={0} max={99} value={p.milesFromMetropolis} onChange={(e) => setP({ ...p, milesFromMetropolis: +e.target.value })} /></label>
        </section>
        <section className="card result">
          <h2>Predicted salary</h2>
          <div className="big">{salary === null ? "…" : `$${salary.toFixed(0)}k`}</div>
          <p className="muted">How this posting's pay grows with experience (0 to 24 years)</p>
          <svg viewBox="-4 -4 308 128" className="curve">
            <path d={path} fill="none" stroke="#818cf8" strokeWidth="3" />
            {curve.length > 0 && <circle cx={(p.yearsExperience / 24) * 300} cy={(() => { const lo = Math.min(...curve) - 5, hi = Math.max(...curve) + 5; return 120 - ((curve[p.yearsExperience] - lo) / (hi - lo)) * 120; })()} r="6" fill="#fbbf24" />}
          </svg>
          {metrics && <p className="muted small">Model: {metrics.model} · holdout MAE ${metrics.mae}k · R² {metrics.r2}</p>}
        </section>
      </div>
      {ins && (
        <section className="card">
          <div className="tabs">
            <h2>Average salary in training data by</h2>
            {["jobType", "industry", "degree", "major"].map((k) => <button key={k} className={chart === k ? "on" : ""} onClick={() => setChart(k)}>{nice(k)}</button>)}
          </div>
          <BarChart data={ins[chart]} highlight={p[chart as keyof Posting] as string} />
        </section>
      )}
    </div>
  );
}
