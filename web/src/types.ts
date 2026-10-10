export type Level = 'PASS' | 'REVIEW' | 'FAIL'

/** v1.1: the literature passage that supports a recommended action, or null when none was verified. */
export interface Basis {
  chunk_id?: string
  title?: string
  url?: string
  license?: string
  excerpt?: string
}
export interface Cause {
  label: string
  note?: string
  metric?: string
  value?: number
  z?: number
  action?: string
  route?: string
  similarity?: number
  basis?: Basis | null
}
export interface Thresholds {
  review?: number
  fail?: number
  source?: string
}
export interface CultureState {
  p_bad?: number
  state?: string
  k?: number
  bank?: number
}
/** v1.1 additions to a QC result. Every key is optional so older data still renders. */
export interface QcExtras {
  route?: string
  attribution?: string
  thresholds?: Thresholds
  reference?: string
  culture_state?: CultureState | null
}
export interface QC extends QcExtras {
  tool_call_id: string
  qc_score: number
  level: Level
  density_state?: string
  metrics?: Record<string, number>
  zbad?: Record<string, number>
  percentiles?: Record<string, number>
  causes?: Cause[]
  latency_ms?: number
}
export interface Evidence {
  chunk_id: string
  title: string
  url?: string
  license?: string
  excerpt?: string
}
export interface Card {
  level: Level
  qc_score: number
  primary_cause?: string
  action?: string
  claims?: string[]
  blocked?: string[]
  evidence?: Evidence[]
  disclaimer?: string
}
export type ClaimStatus = 'supported' | 'unsupported' | 'abstain' | 'refused'
export interface LedgerClaim {
  claim_id: string
  text: string
  type?: string
  evidence?: string[]
  status: ClaimStatus
  reason?: string
}
export interface Ledger {
  claims: LedgerClaim[]
  n_supported?: number
  n_unsupported?: number
  grounded_rate?: number
}
export interface DraftClaim {
  text: string
  type?: string
  evidence?: string[]
}
export interface TraceStep {
  step: number
  tool: string
  latency_ms: number
  tokens?: number
}
export interface Sample {
  id: string
  cell_line?: string
  time_bin?: string
  expert_label?: string
  thumb?: string
  variant?: string
  qc: QC
  card?: Card
  ledger?: Ledger
  ungated_draft?: DraftClaim[]
  trace?: TraceStep[]
  usage?: { llm_calls?: number; prompt_tokens?: number; completion_tokens?: number }
}
export interface Meta {
  version?: string
  branch?: string
  submitted_version?: string
}
export interface SiteData {
  meta?: Meta
  generated?: { git_sha?: string; timestamp_utc?: string; config_hash?: string }
  dataset?: {
    name?: string
    citation?: string
    doi?: string
    license?: string
    n_images_used?: number
    cell_lines?: string[]
  }
  samples: Sample[]
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  results?: Record<string, any>
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  vlm_baseline?: Record<string, any> | null
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  llm_sweep?: Record<string, any> | null
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  rule_citations?: Record<string, any> | null
}
