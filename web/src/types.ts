export type Level = 'PASS' | 'REVIEW' | 'FAIL'

export interface Cause {
  label: string
  note?: string
  metric?: string
  value?: number
  z?: number
  action?: string
}
export interface QC {
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
export interface SiteData {
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
}
