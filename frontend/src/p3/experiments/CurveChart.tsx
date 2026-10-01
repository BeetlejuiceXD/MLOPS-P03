import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ExperimentRun } from "../contracts";
import { type CurveMetric, curveRows } from "../experiments";
import { shortHash } from "../format";

const COLORS = ["#2563EB", "#D97706", "#059669", "#DC2626", "#7C3AED", "#0891B2"];

/**
 * D04-02 — Curva de una métrica por época para uno o varios runs, más la misma serie en
 * tabla (accesible y verificable): los puntos son exactamente el `history` de cada run;
 * una época que un run no tiene queda vacía, no interpolada.
 */
export function CurveChart({
  runs,
  metric,
  showTable = false,
}: Readonly<{ runs: readonly ExperimentRun[]; metric: CurveMetric; showTable?: boolean }>) {
  const rows = curveRows(runs, metric);
  return (
    <div className="flex flex-col gap-3">
      <p className="text-xs font-medium text-ink">{metric}</p>
      <div className="h-64 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={rows} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#E7E5E1" />
            <XAxis dataKey="epoch" tick={{ fontSize: 12 }} allowDecimals={false} />
            <YAxis tick={{ fontSize: 12 }} width={48} domain={["auto", "auto"]} />
            <Tooltip contentStyle={{ borderRadius: 12, fontSize: 12 }} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            {runs.map((run, index) => (
              <Line
                key={run.run_id}
                type="linear"
                dataKey={run.run_id}
                name={shortHash(run.run_id)}
                stroke={COLORS[index % COLORS.length]}
                dot
                connectNulls={false}
                isAnimationActive={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
      {showTable && (
        <details className="text-xs">
          <summary className="cursor-pointer text-ink-muted">Ver valores de {metric}</summary>
          <table data-testid="curve-table" className="mt-2 w-full text-left">
            <thead className="text-ink-muted">
              <tr>
                <th className="px-2 py-1">Época</th>
                {runs.map((run) => (
                  <th key={run.run_id} className="px-2 py-1 font-mono">
                    {shortHash(run.run_id)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.epoch} className="border-t border-border">
                  <td className="px-2 py-1">{row.epoch}</td>
                  {runs.map((run) => {
                    const value = row[run.run_id];
                    return (
                      <td key={run.run_id} className="px-2 py-1 font-mono">
                        {value === null || value === undefined ? "—" : value.toFixed(4)}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </details>
      )}
    </div>
  );
}
