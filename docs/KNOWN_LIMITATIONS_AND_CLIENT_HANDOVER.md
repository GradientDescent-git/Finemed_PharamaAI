# FineMed Pharma AI — Known Limitations & Operational Handover Guide

This document outlines key technical limitations, operational guarantees, and configuration requirements for FineMed Pharma AI.

---

## 1. AI Assistant & Conversational Analytics Disclaimer

- **Operational Guidance**: The embedded LLM conversational service (`POST /chat`) converts natural language queries into grounded time-series analytics and model selection metadata.
- **Human-in-the-Loop Safeguard**: Automated demand predictions and AI recommendations are designed to assist supply chain planners. High-stakes inventory purchasing and medicine reordering decisions must be reviewed and approved by qualified pharmaceutical operations managers.

---

## 2. Outbound Notification Transport Setup

- **Default Behavior**: Structured logging (`logger.error`, `logger.info`) is active out of the box for all pipeline stage events and failure alerts.
- **Production Email & Webhook Dispatch**: Outbound email (SMTP) and webhook (Slack/Teams) alerts require explicit transport credentials in `.env.production`:
  ```env
  NOTIFICATION_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
  NOTIFICATION_SMTP_HOST=smtp.gmail.com
  NOTIFICATION_SMTP_PORT=587
  NOTIFICATION_SMTP_USER=alerts@yourdomain.com
  NOTIFICATION_SMTP_PASSWORD=your-app-password
  NOTIFICATION_EMAIL_FROM=alerts@yourdomain.com
  NOTIFICATION_EMAIL_TO=supplychain-alerts@yourdomain.com
  ```
- **Service SLA**: When SMTP or Webhook details are set, P1 pipeline stage failures trigger immediate outbound alert messages.

---

## 3. Post-Deployment Evaluation & Temporal Window Overlap

- **Temporal Window Dependency**: Forecast evaluation requires actual demand data covering the forecast window.
- **Pending Actuals Behavior**: Evaluating a forecast window (e.g. June 2026) before historical actuals for that window are uploaded yields `has_overlap=False` and `status="PENDING_ACTUALS"`.
- **Honest Metrics Guarantee**: The pipeline explicitly distinguishes pending evaluations from 0.0% WAPE, ensuring un-evaluated forecasts are never reported as perfect forecasts.

---

## 4. PostgreSQL Database Prerequisites

- **Authentication Guard**: Pipeline database loading (`run_database.py`) executes an immediate startup connection test. If PostgreSQL credentials in `.env.production` fail or the database host is unreachable, the runner fails fast with a descriptive `RuntimeError`.
- **Least-Privilege Recommendation**: In production environments, provision a dedicated PostgreSQL role with DDL/DML access restricted to the `warehouse` schema.
