"""
Input Guardrail
Checks user inputs for safety violations.
"""

from typing import Dict, Any, List
import re


class InputGuardrail:
    """
    Guardrail for checking input safety.

    TODO: YOUR CODE HERE
    - Integrate with Guardrails AI or NeMo Guardrails
    - Define validation rules
    - Implement custom validators
    - Handle different types of violations
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize input guardrail.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        safety_config = config.get("safety", config)
        system_config = config.get("system", {})

        self.topic = system_config.get("topic", "HCI Research")
        self.min_length = safety_config.get("min_query_length", 10)
        self.max_length = safety_config.get("max_query_length", 1500)
        self.allowed_topics = [
            "hci",
            "human computer interaction",
            "user experience",
            "ux",
            "ui",
            "usability",
            "accessibility",
            "interaction design",
            "human-centered",
            "human centered",
            "design research",
            "ai",
            "education",
            "healthcare",
            "mobile",
            "voice",
            "ar",
            "vr",
            "explainable ai",
        ]
        self.harmful_patterns = [
            r"\b(build|make|create|write)\b.{0,30}\b(malware|ransomware|virus|keylogger)\b",
            r"\b(hack|exploit|bypass|breach|phish|ddos)\b",
            r"\b(weapon|bomb|poison)\b",
            r"\b(steal|dox|harass|stalk)\b",
        ]
        self.personal_attack_patterns = [
            r"\bidiot\b",
            r"\bstupid\b",
            r"\bworthless\b",
            r"\bkill yourself\b",
        ]
        self.injection_patterns = [
            r"ignore (all|any|the) previous instructions",
            r"forget (all|everything)",
            r"reveal (your )?(system|developer) prompt",
            r"act as (system|developer|root)",
            r"jailbreak",
            r"sudo",
            r"system:",
            r"developer:",
        ]

    def validate(self, query: str) -> Dict[str, Any]:
        """
        Validate input query.

        Args:
            query: User input to validate

        Returns:
            Validation result

        TODO: YOUR CODE HERE
        - Implement validation logic
        - Check for toxic language
        - Check for prompt injection attempts
        - Check query length and format
        - Check for off-topic queries
        """
        normalized_query = query.strip()
        lowered_query = normalized_query.lower()
        violations = []

        if len(normalized_query) < self.min_length:
            violations.append({
                "validator": "length",
                "reason": "Query too short",
                "severity": "medium",
                "category": "invalid_input",
            })

        if len(normalized_query) > self.max_length:
            violations.append({
                "validator": "length",
                "reason": "Query too long",
                "severity": "medium"
            })

        violations.extend(self._check_toxic_language(lowered_query))
        violations.extend(self._check_prompt_injection(lowered_query))
        violations.extend(self._check_relevance(lowered_query))

        blocking_severities = {"high"}
        should_block = any(v.get("severity") in blocking_severities for v in violations)

        return {
            "valid": len(violations) == 0 or not should_block,
            "violations": violations,
            "action": "refuse" if should_block else ("warn" if violations else "allow"),
            "sanitized_input": normalized_query,
        }

    def _check_toxic_language(self, text: str) -> List[Dict[str, Any]]:
        """
        Check for toxic/harmful language.

        TODO: YOUR CODE HERE
        Suggested implementation:
        - Use a moderation API, Guardrails validator, or keyword/rule-based classifier
        - Return a list of violations with validator name, reason, and severity
        - Mark clearly unsafe requests as high severity
        """
        violations = []

        for pattern in self.harmful_patterns:
            if re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL):
                violations.append({
                    "validator": "harmful_content",
                    "reason": "Potentially harmful or abusive request detected",
                    "severity": "high",
                    "category": "harmful_content",
                })
                break

        for pattern in self.personal_attack_patterns:
            if re.search(pattern, text, flags=re.IGNORECASE):
                violations.append({
                    "validator": "personal_attacks",
                    "reason": "Personal attack or abusive language detected",
                    "severity": "high",
                    "category": "personal_attacks",
                })
                break

        return violations

    def _check_prompt_injection(self, text: str) -> List[Dict[str, Any]]:
        """
        Check for prompt injection attempts.

        TODO: YOUR CODE HERE
        Suggested implementation:
        - Detect phrases like \"ignore previous instructions\",
        #   attempts to reveal system prompts, or role-confusion attacks
        - Consider whether the result should block the request or sanitize it
        """
        violations = []
        # Check for common prompt injection patterns
        injection_patterns = [
            "ignore previous instructions",
            "disregard",
            "forget everything",
            "system:",
            "sudo",
        ]

        for pattern in injection_patterns:
            if pattern.lower() in text.lower():
                violations.append({
                    "validator": "prompt_injection",
                    "reason": f"Potential prompt injection: {pattern}",
                    "severity": "high",
                    "category": "prompt_injection",
                })

        return violations

    def _check_relevance(self, query: str) -> List[Dict[str, Any]]:
        """
        Check if query is relevant to the system's purpose.

        TODO: YOUR CODE HERE
        Suggested implementation:
        - Compare the query to the configured topic in config.yaml
        - Use keyword heuristics or an LLM classifier
        - Return low/medium severity violations for off-topic requests
        """
        violations = []
        if any(keyword in query for keyword in self.allowed_topics):
            return violations

        violations.append({
            "validator": "topic_relevance",
            "reason": (
                f"Query appears outside the configured topic area ({self.topic}). "
                "This assistant is intended for HCI-related research questions."
            ),
            "severity": "medium",
            "category": "off_topic_queries",
        })
        return violations
