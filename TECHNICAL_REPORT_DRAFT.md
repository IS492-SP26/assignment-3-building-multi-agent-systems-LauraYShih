# Building and Evaluating a Multi-Agent Research System for HCI Topics

## Abstract

In this project, I built a multi-agent research assistant for HCI-related questions using AutoGen, external search tools, safety guardrails, and LLM-as-a-Judge evaluation. The system supports both a CLI and a Streamlit web interface, and it is designed to make the research process more transparent by showing agent traces, retrieved sources, and safety outcomes. The main workflow uses four roles: Planner, Researcher, Writer, and Critic. Web evidence is collected through Tavily, and academic evidence is requested from Semantic Scholar when available. Because the model and API setup was not always stable during testing, I also added a fallback orchestration path that keeps the same multi-agent structure while improving reliability. In addition, I implemented both input and output guardrails to detect harmful requests, prompt injection attempts, off-topic queries, PII, and unsafe output content. For evaluation, I used two judge perspectives and multiple scoring criteria. Overall, the system can generate grounded and inspectable HCI research answers, although it is still limited by API rate limits and model instability.

## 1. System Design and Implementation

My goal for this project was to build something that felt more like a real research workflow than a one-shot chatbot response. Instead of asking a single model to do everything at once, I split the task into multiple roles so each agent would have a clearer responsibility. This also made it easier to show the grader how the system works internally, which is important for this assignment.

The main architecture uses four agents:

- `Planner`: breaks the query into a short research plan
- `Researcher`: gathers and organizes evidence
- `Writer`: produces the final response with citations
- `Critic`: checks the answer for quality and grounding

The intended workflow is `Planner -> Researcher -> Writer -> Critic`. I used AutoGen as the main orchestration framework because it already supports multi-agent chat patterns and tool integration. At the same time, I wanted the system to remain usable even when the full AutoGen exchange was unreliable. Because of that, the orchestrator also includes a fallback path that still produces planner, researcher, writer, and critic outputs when the full group chat does not return a useful result.

The orchestrator does a few important things before the agents even start talking. First, it runs the input safety checks. Then it prefetches a small amount of evidence from the retrieval tools and passes that evidence into the conversation. I found that this step helped reduce failures from the model and made it easier to preserve real source URLs in the final output.

The tool layer is split into modular files under `src/tools/`:

- `web_search.py` uses Tavily or Brave for general web results
- `paper_search.py` uses Semantic Scholar for academic papers
- `citation_tool.py` handles citation formatting support

One practical issue I ran into was that the original Semantic Scholar client call could hang for too long. I changed that implementation to use a direct HTTP request with a timeout so the system would fail faster and keep the UI responsive. That was one of the biggest engineering fixes during development.

The result returned by the orchestrator includes more than just the final answer. It also includes:

- conversation history
- extracted citations
- structured source metadata
- safety metadata
- agent trace metadata

This made it easy to surface the system behavior in the UI. In the Streamlit app, the user can see the answer, citations, sources, agent traces, and safety events. I also exported a representative sample run to `docs/sample_run.md` so there is at least one end-to-end artifact in the repository.

Overall, I would describe the final system as a hybrid design. It still uses AutoGen as the main multi-agent framework, but it also includes fallback logic so the assignment requirements can still be demonstrated when model/tool behavior is unstable.

## 2. Safety Design

Safety is handled through a `SafetyManager` that coordinates both input and output guardrails. I wanted safety to be part of the normal workflow instead of something added at the very end.

### Input Guardrails

The input guardrail checks for:

- harmful or abusive requests
- personal attacks
- prompt injection attempts
- off-topic queries
- invalid input length

The implementation is mostly rule-based. It normalizes the query, checks for suspicious phrases like "ignore previous instructions," and looks for harmful or abusive intent using keyword and pattern matching. It also checks whether the query seems relevant to the configured topic area, which in this project is HCI research.

If a request is clearly unsafe, the system refuses it. If it is borderline or only weakly relevant, the system can warn or mark it without fully blocking it. This is a simpler approach than using a dedicated moderation model, but it is easy to inspect and explain.

### Output Guardrails

The output guardrail checks for:

- PII such as emails, phone numbers, and SSNs
- harmful or dangerous output
- biased or discriminatory phrasing
- basic grounding inconsistencies relative to available sources

The output guardrail can either allow, sanitize, or refuse the generated text depending on what it finds. For example, PII can be redacted, while more severe unsafe content can trigger a refusal.

### Safety Logging and UI Communication

Every safety event includes:

- timestamp
- whether the check was for input or output
- whether the content was safe
- the action taken
- short violation metadata

These events are exposed in the Streamlit interface, which helps satisfy the assignment requirement that users should be able to tell when content was refused or sanitized and why. The policy categories currently represented in code include:

- `harmful_content`
- `personal_attacks`
- `misinformation`
- `off_topic_queries`
- `prompt_injection`
- `pii`

The biggest limitation here is that the guardrails are heuristic. They work well enough for a course project and they are easy to explain in the report, but they are still not as strong as a production moderation pipeline.

## 3. Evaluation Setup and Results

The evaluation dataset is stored in `data/example_queries.json`. It includes 10 HCI-related prompts across different topics such as explainable AI, accessibility, AR usability, AI in education, visualization, and cross-cultural design. I used this file so the evaluation would not depend on only one kind of query.

The evaluation pipeline is implemented in `src/evaluation/judge.py` and `src/evaluation/evaluator.py`. The scoring criteria are defined in `config.yaml`:

- relevance
- evidence quality
- factual accuracy
- safety compliance
- clarity

To meet the assignment requirement for multiple judging perspectives, I used two judge roles:

1. `research_rigor`, which focuses more on grounding, completeness, and evidence quality
2. `usability_reader`, which focuses more on clarity, usefulness, and safety communication

Scores are aggregated into criterion-level results and an overall weighted score. I also added a lightweight fallback evaluation path so I could still test the pipeline even when a live judge call was unavailable.

In my live demo runs, the system was able to answer the accessibility query with visible sources, citations, and multi-agent traces in the UI. That is the main result I would highlight in the report, because it demonstrates that the system works end to end and satisfies the main rubric categories for orchestration, transparency, and safety communication.

At the same time, the evaluation setup exposed several practical problems:

- Semantic Scholar sometimes returned rate-limit errors
- Groq tool-calling was not stable enough for the full AutoGen workflow
- larger prompts sometimes exceeded token-per-minute limits

These issues pushed me to compress prompts, reduce message budgets, prefetch evidence, and add a fallback orchestration path. So in that sense, the evaluation was useful not only for scoring the system, but also for identifying which parts of the architecture were too fragile.

## 4. Discussion and Limitations

The biggest strength of the final system is that it is transparent. Even when the underlying model setup is imperfect, the app still shows the user what the planner, researcher, writer, and critic did, which sources were used, and whether any safety rules were triggered. That makes the system much easier to inspect than a single black-box answer.

Another strength is that the safety logic is integrated directly into the orchestration path. Unsafe input is checked before generation, unsafe output is checked after generation, and safety metadata is visible in the UI. I think this is important for a research assistant because trust depends not just on the final answer, but also on how the system handles risky inputs and outputs.

The main weakness is reliability. In practice, the full AutoGen workflow depended heavily on external APIs and model behavior. The Groq model was fast and easy to access, but it was not always reliable for multi-agent tool-calling. Semantic Scholar also introduced rate-limit issues, which meant that some runs leaned more on web sources than academic papers.

There are also limits to the current safety and evaluation approaches. The guardrails are mostly rule-based, so they can miss subtle cases. The judge framework is helpful, but LLM-as-a-Judge is still noisy and depends on prompt wording and provider behavior. Because of this, I would treat the evaluation results as useful signals rather than perfect ground truth.

If I had more time, I would improve the project in a few ways:

- move to a more stable model/provider for AutoGen tool-calling
- cache search results and judge results
- strengthen the grounding checks beyond simple URL matching
- add a stronger moderation or validation layer
- make the UI show evaluation summaries more directly

Even with these limitations, I think the final system meets the core goals of the assignment. It demonstrates a real multi-agent workflow, exposes the internal process to the user, integrates safety checks, and includes an evaluation pipeline with multiple judge perspectives.

## References

AutoGen documentation. <https://microsoft.github.io/autogen/>

Tavily API documentation. <https://docs.tavily.com/>

Semantic Scholar API documentation. <https://api.semanticscholar.org/>

Guardrails AI documentation. <https://docs.guardrailsai.com/>

NeMo Guardrails documentation. <https://docs.nvidia.com/nemo/guardrails/>
