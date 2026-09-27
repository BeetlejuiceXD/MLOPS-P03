/** D01-05 — Formatos cortos compartidos por las páginas P3. */
export const shortHash = (value: string) => value.slice(0, 12);
export const percent = (value: number) => `${(value * 100).toFixed(2)}%`;
export const dateTime = (value: string | null) =>
  value === null ? "—" : new Date(value).toLocaleString("es-MX");
