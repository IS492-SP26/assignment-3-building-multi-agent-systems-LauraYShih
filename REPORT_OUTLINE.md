# Technical Report Outline

Use this as the starting point for the 3-4 page single-column report required by the assignment.

## Abstract

Briefly describe:

- the goal of the system
- the agent architecture
- the safety guardrails
- the evaluation method
- the main findings

Target length: about 150 words.

## 1. System Design and Implementation

Explain:

- why you chose AutoGen
- the roles of the `Planner`, `Researcher`, `Writer`, and `Critic`
- how the orchestrator moves information between agents
- which external tools were used:
  - Tavily or Brave for web search
  - Semantic Scholar for paper search
- how citations and traces are extracted for the UI

Suggested evidence to include:

- one workflow diagram
- one example of a planner output
- one example of citations shown in the interface

## 2. Safety Design

Describe the safety pipeline:

- input guardrail categories:
  - harmful content
  - personal attacks
  - prompt injection
  - off-topic queries
- output guardrail categories:
  - PII
  - unsafe content
  - bias
  - grounding / misinformation checks
- how violations are handled:
  - allow
  - warn
  - sanitize
  - refuse
- how safety events are logged and surfaced in the UI

Include at least one blocked or sanitized example.

## 3. Evaluation Setup and Results

Document:

- the evaluation dataset in `data/example_queries.json`
- why the queries are diverse and HCI-relevant
- the judge criteria from `config.yaml`
- the two judging perspectives:
  - `research_rigor`
  - `usability_reader`

Recommended table/figure content:

- average score by criterion
- best-performing query
- worst-performing query
- short qualitative error analysis

## 4. Discussion and Limitations

Discuss:

- what worked well in the multi-agent design
- where the system still fails
- limitations of the safety heuristics
- limitations of search quality and retrieval coverage
- dependence on external APIs
- how the judge setup could be improved

## References

Use APA style.

Likely citations:

- AutoGen documentation
- Tavily API documentation
- Semantic Scholar API documentation
- Guardrails or NeMo Guardrails documentation
- any HCI papers you cite in the report
