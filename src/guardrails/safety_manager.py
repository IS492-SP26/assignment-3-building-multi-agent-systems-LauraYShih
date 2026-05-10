"""
Safety Manager
Coordinates safety guardrails and logs safety events.
"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import json

from src.guardrails.input_guardrail import InputGuardrail
from src.guardrails.output_guardrail import OutputGuardrail


class SafetyManager:
    """
    Manages safety guardrails for the multi-agent system.

    TODO: YOUR CODE HERE
    - Integrate with Guardrails AI or NeMo Guardrails
    - Define safety policies
    - Implement logging of safety events
    - Handle different violation types with appropriate responses
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize safety manager.

        Args:
            config: Safety configuration
        """
        self.config = config
        self.safety_config = config.get("safety", config)
        self.logging_config = config.get("logging", {})
        self.enabled = self.safety_config.get("enabled", True)
        self.log_events = self.safety_config.get("log_events", True)
        self.logger = logging.getLogger("safety")

        # Safety event log
        self.safety_events: List[Dict[str, Any]] = []

        # Prohibited categories
        self.prohibited_categories = self.safety_config.get("prohibited_categories", [
            "harmful_content",
            "personal_attacks",
            "misinformation",
            "off_topic_queries"
        ])

        # Violation response strategy
        self.on_violation = self.safety_config.get("on_violation", {})
        self.safety_log_file = self.logging_config.get("safety_log", "logs/safety_events.log")
        self.input_guardrail = InputGuardrail(config)
        self.output_guardrail = OutputGuardrail(config)

    def check_input_safety(self, query: str) -> Dict[str, Any]:
        """
        Check if input query is safe to process.

        Args:
            query: User query to check

        Returns:
            Dictionary with 'safe' boolean and optional 'violations' list

        TODO: YOUR CODE HERE
        - Implement guardrail checks
        - Detect harmful/inappropriate content
        - Detect off-topic queries
        - Return detailed violation information
        """
        if not self.enabled:
            return {"safe": True, "action": "allow", "query": query, "violations": []}

        validation = self.input_guardrail.validate(query)
        is_safe = validation.get("valid", True)
        violations = validation.get("violations", [])
        action = validation.get("action", "allow")

        if violations and self.log_events:
            self._log_safety_event("input", query, violations, is_safe, action=action)

        return {
            "safe": is_safe,
            "violations": violations,
            "action": action,
            "query": validation.get("sanitized_input", query),
            "message": self.on_violation.get(
                "message",
                "I cannot process this request due to safety policies."
            ) if not is_safe else "",
        }

    def check_output_safety(
        self,
        response: str,
        sources: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Check if output response is safe to return.

        Args:
            response: Generated response to check
            sources: Optional source metadata used by output validation

        Returns:
            Dictionary with 'safe' boolean and optional 'violations' list

        TODO: YOUR CODE HERE
        - Implement output guardrail checks
        - Detect harmful content in responses
        - Detect potential misinformation
        - Sanitize or redact unsafe content
        """
        if not self.enabled:
            return {"safe": True, "response": response}
        validation = self.output_guardrail.validate(response, sources)
        is_safe = validation.get("valid", True)
        violations = validation.get("violations", [])
        action = validation.get("action", "allow")
        output_response = validation.get("sanitized_output", response)

        if violations and self.log_events:
            self._log_safety_event("output", response, violations, is_safe, action=action)

        return {
            "safe": is_safe,
            "violations": violations,
            "action": action,
            "response": output_response,
        }

    def _sanitize_response(self, response: str, violations: List[Dict[str, Any]]) -> str:
        """
        Sanitize response by removing or redacting unsafe content.
        """
        redacted = response
        for violation in violations:
            for match in violation.get("matches", []):
                redacted = redacted.replace(match, "[REDACTED]")
        return redacted

    def _log_safety_event(
        self,
        event_type: str,
        content: str,
        violations: List[Dict[str, Any]],
        is_safe: bool,
        action: str = "allow",
    ):
        """
        Log a safety event.

        Args:
            event_type: "input" or "output"
            content: The content that was checked
            violations: List of violations found
            is_safe: Whether content passed safety checks
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "safe": is_safe,
            "action": action,
            "violations": violations,
            "content_preview": content[:100] + "..." if len(content) > 100 else content
        }

        self.safety_events.append(event)
        self.logger.warning(f"Safety event: {event_type} - safe={is_safe}")

        # Write to safety log file if configured
        log_file = self.safety_log_file
        if log_file and self.log_events:
            try:
                with open(log_file, "a") as f:
                    f.write(json.dumps(event) + "\n")
            except Exception as e:
                self.logger.error(f"Failed to write safety log: {e}")

    def get_safety_events(self) -> List[Dict[str, Any]]:
        """Get all logged safety events."""
        return self.safety_events

    def get_safety_stats(self) -> Dict[str, Any]:
        """
        Get statistics about safety events.

        Returns:
            Dictionary with safety statistics
        """
        total = len(self.safety_events)
        input_events = sum(1 for e in self.safety_events if e["type"] == "input")
        output_events = sum(1 for e in self.safety_events if e["type"] == "output")
        violations = sum(1 for e in self.safety_events if not e["safe"])

        return {
            "total_events": total,
            "input_checks": input_events,
            "output_checks": output_events,
            "violations": violations,
            "violation_rate": violations / total if total > 0 else 0
        }

    def clear_events(self):
        """Clear safety event log."""
        self.safety_events = []
