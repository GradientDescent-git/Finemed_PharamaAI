from finemed_ai.forecast_intelligence.conversation_orchestrator import (
    ForecastConversationOrchestrator,
)


def test_greeting_does_not_destroy_medicine_context():
    service = ForecastConversationOrchestrator()

    first = service.ask(
        "What is the forecast for Otacare?"
    )

    assert first["resolved_medicine"] == (
        "OTACARE EAR DROPS 5ML"
    )

    greeting = service.ask("hello")

    assert greeting["action"] == "greeting"

    context = service.get_context()

    assert context["medicine_name"] == (
        "OTACARE EAR DROPS 5ML"
    )

    follow_up = service.ask(
        "Which model predicted it?"
    )

    assert follow_up["resolved_medicine"] == (
        "OTACARE EAR DROPS 5ML"
    )


def test_help_does_not_destroy_medicine_context():
    service = ForecastConversationOrchestrator()

    service.ask(
        "What is the forecast for Otacare?"
    )

    service.ask(
        "What can you help me with?"
    )

    follow_up = service.ask(
        "Why was that model selected?"
    )

    assert follow_up["resolved_medicine"] == (
        "OTACARE EAR DROPS 5ML"
    )


def test_out_of_scope_does_not_destroy_context():
    service = ForecastConversationOrchestrator()

    service.ask(
        "What is the forecast for Otacare?"
    )

    service.ask(
        "What is the weather today?"
    )

    follow_up = service.ask(
        "What about the P90?"
    )

    assert follow_up["resolved_medicine"] == (
        "OTACARE EAR DROPS 5ML"
    )


def test_unknown_medicine_does_not_replace_valid_context():
    service = ForecastConversationOrchestrator()

    service.ask(
        "What is the forecast for Otacare?"
    )

    unknown = service.ask(
        "What is the forecast for XYZ_UNKNOWN?"
    )

    assert unknown["data"]["found"] is False

    context = service.get_context()

    assert context["medicine_name"] == (
        "OTACARE EAR DROPS 5ML"
    )

    follow_up = service.ask(
        "Which model predicted it?"
    )

    assert follow_up["resolved_medicine"] == (
        "OTACARE EAR DROPS 5ML"
    )


def test_new_resolved_medicine_replaces_previous_context():
    service = ForecastConversationOrchestrator()

    service.ask(
        "What is the forecast for Otacare?"
    )

    second = service.ask(
        "What is the forecast for Keelac?"
    )

    assert second["data"]["found"] is True

    context = service.get_context()

    assert context["medicine_name"] == (
        second["resolved_medicine"]
    )


def test_follow_up_without_context_is_not_invented():
    service = ForecastConversationOrchestrator()

    result = service.ask(
        "Which model predicted it?"
    )

    assert result["action"] == "model_info"

    assert (
        result["data"].get("found") is False
        or result["data"].get("reason")
        in {
            "medicine_not_resolved",
            "insufficient_context",
        }
    )