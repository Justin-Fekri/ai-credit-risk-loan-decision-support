import { FEATURE_LABELS, PURPOSE_LABELS } from "./copy";

export function money(value: number) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(value);
}

export function pct(value: number, digits = 1) {
  return `${(value * 100).toFixed(digits)}%`;
}

export function featureLabel(key: string) {
  return FEATURE_LABELS[key] ?? key.replaceAll("_", " ");
}

export function formatFeature(key: string, value: string | number) {
  if (typeof value === "number") {
    if (["annual_income", "loan_amount", "monthly_debt"].includes(key)) {
      return money(value);
    }
    if (["debt_to_income", "revolving_utilization"].includes(key)) {
      return pct(value);
    }
    if (["employment_length_years", "credit_history_length_years"].includes(key)) {
      return `${value} yrs`;
    }
  }
  if (key === "loan_purpose") {
    return PURPOSE_LABELS[String(value)] ?? String(value).replaceAll("_", " ");
  }
  return String(value).replaceAll("_", " ");
}

export function riskCopy(band: string) {
  return {
    low: "Low",
    moderate: "Moderate",
    elevated: "Elevated",
    severe: "Severe",
  }[band] ?? band;
}

export function recCopy(rec: string) {
  return {
    approve: "Recommend approve",
    refer: "Refer to underwriter",
    decline: "Recommend decline",
  }[rec] ?? rec;
}
