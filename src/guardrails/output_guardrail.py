"""
Output Guardrail
Checks system outputs for safety violations.
"""

from typing import Dict, Any, List
import re


class OutputGuardrail:
    """
    Guardrail for checking output safety.

    TODO: YOUR CODE HERE
    - Integrate with Guardrails AI or NeMo Guardrails
    - Check for harmful content in responses
    - Verify factual consistency
    - Detect potential misinformation
    - Remove PII (personal identifiable information)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize output guardrail.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        safety_config = config.get("safety", config)
        self.block_on_harmful = safety_config.get("block_harmful_output", True)
        self.redact_pii = safety_config.get("redact_pii", True)
        self.response_message = safety_config.get(
            "on_violation",
            {},
        ).get("message", "I cannot provide that response due to safety policies.")

    def validate(self, response: str, sources: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate output response.

        Args:
            response: Generated response to validate
            sources: Optional list of sources used (for fact-checking)

        Returns:
            Validation result

        TODO: YOUR CODE HERE
        - Implement validation logic
        - Check for harmful content
        - Check for PII
        - Verify claims against sources
        - Check for bias
        """
        violations = []

        pii_violations = self._check_pii(response)
        violations.extend(pii_violations)

        harmful_violations = self._check_harmful_content(response)
        violations.extend(harmful_violations)

        bias_violations = self._check_bias(response)
        violations.extend(bias_violations)

        if sources:
            consistency_violations = self._check_factual_consistency(response, sources)
            violations.extend(consistency_violations)

        should_block = any(
            violation.get("severity") == "high" and violation.get("validator") != "pii"
            for violation in violations
        )
        sanitized_output = self._sanitize(response, violations) if violations else response
        if should_block:
            sanitized_output = self.response_message

        return {
            "valid": len(violations) == 0 or not should_block,
            "violations": violations,
            "action": "refuse" if should_block else ("sanitize" if violations else "allow"),
            "sanitized_output": sanitized_output,
        }

    def _check_pii(self, text: str) -> List[Dict[str, Any]]:
        """
        Check for personally identifiable information.

        TODO: YOUR CODE HERE
        Suggested implementation:
        - Expand regex checks for emails, phone numbers, SSNs, addresses, etc.
        - Use a stronger PII detection library if desired
        - Return violation metadata needed for redaction
        """
        violations = []

        # Simple regex patterns for common PII
        patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
        }

        for pii_type, pattern in patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                violations.append({
                    "validator": "pii",
                    "pii_type": pii_type,
                    "reason": f"Contains {pii_type}",
                    "severity": "high",
                    "matches": matches,
                    "category": "pii",
                })

        return violations

    def _check_harmful_content(self, text: str) -> List[Dict[str, Any]]:
        """
        Check for harmful or inappropriate content.

        TODO: YOUR CODE HERE
        Suggested implementation:
        - Detect unsafe instructions, hateful content, or violent guidance
        - Use a moderation model, guardrail validator, or rule-based policy check
        - Return severity levels so the caller knows whether to refuse or sanitize
        """
        violations = []

        # Placeholder - should use proper toxicity detection
        harmful_keywords = [
            "how to build a bomb",
            "how to hack",
            "kill them",
            "violent attack",
            "bypass the law",
        ]
        for keyword in harmful_keywords:
            if keyword in text.lower():
                violations.append({
                    "validator": "harmful_content",
                    "reason": f"May contain harmful content: {keyword}",
                    "severity": "high",
                    "category": "harmful_content",
                })

        return violations

    def _check_factual_consistency(
        self,
        response: str,
        sources: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Check if response is consistent with sources.

        TODO: YOUR CODE HERE
        Suggested implementation:
        - Compare claims in the response against the retrieved evidence
        - Verify that citations actually support the statements made
        - Optionally use an LLM-based verifier or a citation-grounding check
        """
        violations = []
        if not sources:
            return violations

        response_lower = response.lower()
        cited_urls = {
            source.get("url", "").strip()
            for source in sources
            if isinstance(source, dict) and source.get("url")
        }

        claimed_urls = set(re.findall(r"https?://[^\s<>{}\[\]]+", response))
        missing_urls = claimed_urls - cited_urls
        if missing_urls:
            violations.append({
                "validator": "factual_consistency",
                "reason": "Response cites URLs that were not present in retrieved sources",
                "severity": "medium",
                "category": "misinformation",
                "matches": sorted(missing_urls),
            })

        if "according to the sources" in response_lower and not cited_urls:
            violations.append({
                "validator": "factual_consistency",
                "reason": "Response claims source support but no structured sources were provided",
                "severity": "medium",
                "category": "misinformation",
            })

        return violations

    def _check_bias(self, text: str) -> List[Dict[str, Any]]:
        """
        Check for biased language.

        TODO: YOUR CODE HERE
        Suggested implementation:
        - Look for stereotypes, blanket generalizations, or discriminatory language
        - Decide whether to redact, revise, or refuse the output
        """
        violations = []
        biased_phrases = [
            "all people from",
            "naturally inferior",
            "genetically better",
            "those people are",
        ]

        lowered = text.lower()
        for phrase in biased_phrases:
            if phrase in lowered:
                violations.append({
                    "validator": "bias",
                    "reason": f"Potential biased or discriminatory phrasing detected: {phrase}",
                    "severity": "high",
                    "category": "bias",
                })
                break
        return violations

    def _sanitize(self, text: str, violations: List[Dict[str, Any]]) -> str:
        """
        Sanitize text by removing/redacting violations.

        TODO: YOUR CODE HERE
        Suggested implementation:
        - Redact matched PII spans
        - Replace unsafe sections with placeholder text
        - Optionally return a refusal message for severe violations
        """
        sanitized = text

        # Redact PII
        for violation in violations:
            if violation.get("validator") == "pii":
                for match in violation.get("matches", []):
                    sanitized = sanitized.replace(match, "[REDACTED]")

        for violation in violations:
            if violation.get("validator") == "harmful_content":
                return self.response_message

        return sanitized
