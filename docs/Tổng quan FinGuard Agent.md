# Tổng quan FinGuard Agent

FinGuard Agent là trợ lý tài chính dùng Streamlit và LLM, có các công cụ tính
toán tài chính cùng nhiều lớp bảo vệ đầu vào, đầu ra và dữ liệu cá nhân.

## Trạng thái tích hợp NeMo

- NeMo Guardrails được tích hợp với Nebius Token Factory.
- Model phân loại: `nvidia/Llama-3.1-Nemotron-70B-Instruct`.
- Phân loại prompt injection bằng LLM hoạt động như lớp phòng vệ thứ hai, sau
  các mẫu regex EN và VI chạy cục bộ.
- `pytest tests/test_nemo_guard.py -v` đã báo cáo 5 passed, bao gồm
  `test_obfuscated_injection`.

## Ưu tiên cao

- [x] Xác minh NeMo Guardrails và phân loại prompt injection bằng Nemotron.
- [x] Đóng mặc định khi dịch vụ output moderation gặp lỗi; có thể cấu hình
  fail-open bằng `MODERATION_FAIL_CLOSED=false`.
- [ ] User authentication + JWT
- [ ] Redis-based server-side rate limiting
- [ ] Streaming responses (SSE)
- [ ] Prometheus metrics + Grafana dashboards
- [ ] Multi-language UI (Vietnamese + English)
- [ ] Docker Compose with Redis + Postgres
