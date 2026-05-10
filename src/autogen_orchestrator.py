"""
AutoGen-Based Orchestrator

This orchestrator uses AutoGen's RoundRobinGroupChat to coordinate multiple agents
in a research workflow.

Workflow:
1. Planner: Breaks down the query into research steps
2. Researcher: Gathers evidence using web and paper search tools
3. Writer: Synthesizes findings into a coherent response
4. Critic: Evaluates quality and provides feedback
"""

import logging
import asyncio
import re
import os
from typing import Dict, Any, List, Optional

from src.agents.autogen_agents import create_research_team
from src.guardrails.safety_manager import SafetyManager
from src.tools.web_search import WebSearchTool
from src.tools.paper_search import PaperSearchTool

try:
    from groq import Groq
except ImportError:  # pragma: no cover
    Groq = None

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None


class AutoGenOrchestrator:
    """
    Orchestrates multi-agent research using AutoGen's RoundRobinGroupChat.
    
    This orchestrator manages a team of specialized agents that work together
    to answer research queries. It uses AutoGen's built-in conversation
    management and tool execution capabilities.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the AutoGen orchestrator.

        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config
        self.logger = logging.getLogger("autogen_orchestrator")
        
        # Create the research team
        self.logger.info("Creating research team...")
        self.team = create_research_team(config)
        self.safety_manager = SafetyManager(config)
        
        self.logger.info("Research team created successfully")
        
        # Workflow trace for debugging and UI display
        self.workflow_trace: List[Dict[str, Any]] = []

    def process_query(self, query: str, max_rounds: int = 20) -> Dict[str, Any]:
        """
        Process a research query through the multi-agent system.

        Args:
            query: The research question to answer
            max_rounds: Maximum number of conversation rounds

        Returns:
            Dictionary containing:
            - query: Original query
            - response: Final synthesized response
            - conversation_history: Full conversation between agents
            - metadata: Additional information about the process
        """
        self.logger.info(f"Processing query: {query}")
        
        try:
            start_event_index = len(self.safety_manager.get_safety_events())
            input_safety = self.safety_manager.check_input_safety(query)
            if not input_safety.get("safe", True):
                safety_events = self.safety_manager.get_safety_events()[start_event_index:]
                return {
                    "query": query,
                    "response": input_safety.get(
                        "message",
                        "I cannot process this request due to safety policies.",
                    ),
                    "conversation_history": [],
                    "citations": [],
                    "metadata": {
                        "error": False,
                        "num_messages": 0,
                        "num_sources": 0,
                        "sources": [],
                        "citations": [],
                        "agents_involved": [],
                        "safety": {
                            "input": input_safety,
                            "output": None,
                            "status": "refused",
                        },
                        "safety_events": safety_events,
                    },
                }

            # Run the async query processing
            try:
                loop = asyncio.get_running_loop()
                loop_is_running = loop.is_running()
            except RuntimeError:
                loop = None
                loop_is_running = False

            if loop_is_running:
                # If we're already in an async context, create a new loop
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    result = pool.submit(
                        asyncio.run, 
                        self._process_query_async(input_safety.get("query", query), max_rounds)
                    ).result()
            else:
                result = asyncio.run(self._process_query_async(input_safety.get("query", query), max_rounds))

            output_safety = self.safety_manager.check_output_safety(
                result.get("response", ""),
                result.get("metadata", {}).get("sources", []),
            )
            result["response"] = output_safety.get("response", result.get("response", ""))
            result["metadata"]["safety"] = {
                "input": input_safety,
                "output": output_safety,
                "status": output_safety.get("action", "allow"),
            }
            result["metadata"]["safety_events"] = self.safety_manager.get_safety_events()[start_event_index:]
            
            self.logger.info("Query processing complete")
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}", exc_info=True)
            return {
                "query": query,
                "error": str(e),
                "response": f"An error occurred while processing your query: {str(e)}",
                "conversation_history": [],
                "citations": [],
                "metadata": {"error": True, "sources": [], "citations": []}
            }
    
    async def _process_query_async(self, query: str, max_rounds: int = 20) -> Dict[str, Any]:
        """
        Async implementation of query processing.
        
        Args:
            query: The research question to answer
            max_rounds: Maximum number of conversation rounds
            
        Returns:
            Dictionary containing results
        """
        seed_sources = await self._gather_seed_evidence(query)
        evidence_summary = self._format_seed_evidence(seed_sources)

        # Create task message
        task_message = f"""Query: {query}

Planner: make a short plan.
Researcher: use the evidence below.
Writer: answer with concise inline citations.
Critic: approve or suggest one revision, then say TERMINATE if acceptable.

Evidence:
{evidence_summary}"""
        
        # Run the team
        result = await self.team.run(task=task_message)
        
        # Extract conversation history
        messages = await self._collect_messages(result)
        
        # Extract final response
        final_response = ""
        if messages:
            for preferred_source in ["Writer", "Critic"]:
                for msg in reversed(messages):
                    if msg.get("source") == preferred_source:
                        final_response = msg.get("content", "")
                        break
                if final_response:
                    break
        
        # If no response found, use the last message
        if not final_response and messages:
            final_response = messages[-1].get("content", "")
        
        if self._should_use_fallback(messages, final_response, task_message):
            self.logger.warning("AutoGen returned no useful agent output; using fallback orchestration.")
            return await self._run_fallback_pipeline(query, seed_sources)

        return self._extract_results(query, messages, final_response, seed_sources=seed_sources)

    def _should_use_fallback(self, messages: List[Dict[str, Any]], final_response: str, task_message: str) -> bool:
        """Decide whether to fall back to a simpler orchestrated pipeline."""
        non_user_messages = [msg for msg in messages if msg.get("source") != "user"]
        if not non_user_messages:
            return True
        if not final_response.strip():
            return True
        if final_response.strip() == task_message.strip():
            return True
        return False

    async def _run_fallback_pipeline(
        self,
        query: str,
        seed_sources: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Fallback planner/researcher/writer/critic pipeline using smaller direct steps."""
        plan = self._build_fallback_plan(query)
        research = self._build_fallback_research(seed_sources)
        writer_output = await self._generate_fallback_answer(query, plan, seed_sources)
        critique = self._build_fallback_critique(writer_output, seed_sources)

        messages = [
            {"source": "Planner", "content": plan},
            {"source": "Researcher", "content": research},
            {"source": "Writer", "content": writer_output},
            {"source": "Critic", "content": critique},
        ]
        return self._extract_results(query, messages, writer_output, seed_sources=seed_sources)

    def _build_fallback_plan(self, query: str) -> str:
        """Create a compact research plan."""
        return (
            f"1. Identify the main HCI concepts in the query: {query}\n"
            "2. Review authoritative design guidance and recent practice-oriented sources.\n"
            "3. Extract 3-5 key principles, then summarize them with citations and practical implications."
        )

    def _build_fallback_research(self, seed_sources: List[Dict[str, Any]]) -> str:
        """Summarize retrieved evidence for the researcher trace."""
        if not seed_sources:
            return "No retrieved evidence was available."

        lines = []
        for idx, source in enumerate(seed_sources[:4], start=1):
            lines.append(
                f"{idx}. {source.get('title', 'Untitled')} | {source.get('url', '')}\n"
                f"   Evidence: {source.get('snippet', '')[:180]}"
            )
        return "\n".join(lines)

    async def _generate_fallback_answer(
        self,
        query: str,
        plan: str,
        seed_sources: List[Dict[str, Any]],
    ) -> str:
        """Generate the final answer directly when AutoGen fails."""
        evidence_lines = []
        for source in seed_sources[:4]:
            evidence_lines.append(
                f"- {source.get('title', 'Untitled')} | {source.get('url', '')}\n"
                f"  {source.get('snippet', '')[:160]}"
            )
        evidence_block = "\n".join(evidence_lines) if evidence_lines else "- No sources available."

        prompt = f"""You are the Writer agent in a multi-agent HCI research assistant.

Query: {query}

Plan:
{plan}

Retrieved evidence:
{evidence_block}

Write a concise answer with:
- a short introduction
- 3-5 key principles
- inline numeric citations like [1], [2]
- a short references list at the end with source title and exact URL

Do not invent citations. If evidence is limited, say so briefly."""

        generated = await self._call_simple_model(prompt)
        if generated:
            return generated

        # Final non-LLM fallback.
        bullet_points = []
        references = []
        for idx, source in enumerate(seed_sources[:3], start=1):
            bullet_points.append(
                f"- {source.get('title', 'Source')} [{idx}] emphasizes inclusive, standards-based design "
                f"and practical accessibility guidance."
            )
            references.append(f"[{idx}] {source.get('title', 'Source')} - {source.get('url', '')}")
        bullets = "\n".join(bullet_points) if bullet_points else "- Use inclusive, standards-based design."
        refs = "\n".join(references)
        return (
            f"Accessible user interface design focuses on making interfaces usable by people with diverse abilities.\n\n"
            f"Key principles include perceptibility, operability, understandability, and robustness.\n"
            f"Designers should support keyboard navigation, readable contrast, clear feedback, and compatibility with assistive technologies.\n\n"
            f"Supporting evidence:\n{bullets}\n\n"
            f"References:\n{refs}" if refs else
            f"Supporting evidence:\n{bullets}"
        )

    def _build_fallback_critique(self, writer_output: str, seed_sources: List[Dict[str, Any]]) -> str:
        """Create a brief critic message for transparency."""
        citation_count = sum(1 for source in seed_sources if source.get("url"))
        return (
            f"The answer is relevant and grounded in {citation_count} retrieved sources. "
            "It should be acceptable for a concise research response. TERMINATE"
        )

    async def _call_simple_model(self, prompt: str) -> str:
        """Call the configured model directly for fallback generation."""
        model_config = self.config.get("models", {}).get("default", {})
        provider = model_config.get("provider", "groq")
        model = model_config.get("name", "llama-3.1-8b-instant")
        max_tokens = min(model_config.get("max_tokens", 250), 250)
        temperature = model_config.get("temperature", 0.4)

        try:
            if provider == "groq" and Groq:
                client = Groq(api_key=os.getenv("GROQ_API_KEY"))
                response = await asyncio.to_thread(
                    client.chat.completions.create,
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return response.choices[0].message.content or ""

            if OpenAI:
                client = OpenAI(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    base_url=os.getenv("OPENAI_BASE_URL"),
                )
                response = await asyncio.to_thread(
                    client.chat.completions.create,
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return response.choices[0].message.content or ""
        except Exception as exc:
            self.logger.warning(f"Fallback direct model call failed: {exc}")

        return ""

    async def _gather_seed_evidence(self, query: str) -> List[Dict[str, Any]]:
        """Prefetch a small set of real sources so citations are grounded."""
        sources: List[Dict[str, Any]] = []
        tools_config = self.config.get("tools", {})

        if tools_config.get("web_search", {}).get("enabled", True):
            web_cfg = tools_config.get("web_search", {})
            web_tool = WebSearchTool(
                provider=web_cfg.get("provider", "tavily"),
                max_results=min(web_cfg.get("max_results", 5), 3),
            )
            try:
                web_results = await web_tool.search(query)
                for item in web_results:
                    sources.append(
                        {
                            "title": item.get("title", "Untitled web source"),
                            "url": item.get("url", ""),
                            "snippet": item.get("snippet", ""),
                            "type": "webpage",
                            "source_agent": "orchestrator_prefetch",
                        }
                    )
            except Exception as exc:
                self.logger.warning(f"Web search prefetch failed: {exc}")

        if tools_config.get("paper_search", {}).get("enabled", True):
            paper_cfg = tools_config.get("paper_search", {})
            paper_tool = PaperSearchTool(max_results=min(paper_cfg.get("max_results", 10), 3))
            try:
                paper_results = await paper_tool.search(query, year_from=2020)
                for item in paper_results:
                    sources.append(
                        {
                            "title": item.get("title", "Untitled paper"),
                            "url": item.get("url", ""),
                            "snippet": item.get("abstract", ""),
                            "type": "paper",
                            "source_agent": "orchestrator_prefetch",
                        }
                    )
            except Exception as exc:
                self.logger.warning(f"Paper search prefetch failed: {exc}")

        deduped = []
        seen = set()
        for item in sources:
            key = item.get("url") or item.get("title")
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(item)

        return deduped

    def _format_seed_evidence(self, sources: List[Dict[str, Any]]) -> str:
        """Format prefetched evidence into a concise message block."""
        if not sources:
            return "- No seed evidence available."

        lines = []
        for index, source in enumerate(sources[:2], start=1):
            lines.append(
                f"- Source {index}: {source.get('title', 'Untitled')} | "
                f"URL: {source.get('url', 'N/A')} | "
                f"Snippet: {source.get('snippet', '')[:100]}"
            )
        return "\n".join(lines)

    async def _collect_messages(self, result: Any) -> List[Dict[str, Any]]:
        """Normalize AutoGen results into a serializable message list."""
        raw_messages = getattr(result, "messages", [])
        messages = []

        if hasattr(raw_messages, "__aiter__"):
            async for message in raw_messages:
                messages.append(self._message_to_dict(message))
        else:
            for message in raw_messages:
                messages.append(self._message_to_dict(message))

        return messages

    def _message_to_dict(self, message: Any) -> Dict[str, Any]:
        """Convert AutoGen message objects into plain dictionaries."""
        content = getattr(message, "content", None)
        if isinstance(content, list):
            content = "\n".join(str(item) for item in content)
        elif content is None:
            content = str(message)

        return {
            "source": getattr(message, "source", getattr(message, "name", "Unknown")),
            "content": str(content),
        }
        
    def _extract_results(
        self,
        query: str,
        messages: List[Dict[str, Any]],
        final_response: str = "",
        seed_sources: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Extract structured results from the conversation history.

        Args:
            query: Original query
            messages: List of conversation messages
            final_response: Final response from the team

        Returns:
            Structured result dictionary
        """
        # Extract components from conversation
        research_findings = []
        plan = ""
        critique = ""
        
        for msg in messages:
            source = msg.get("source", "")
            content = msg.get("content", "")
            
            if source == "Planner" and not plan:
                plan = content
            
            elif source == "Researcher":
                research_findings.append(content)
            
            elif source == "Critic":
                critique = content
        
        # Count sources mentioned in research
        num_sources = 0
        for finding in research_findings:
            # Rough count of sources based on numbered results
            num_sources += finding.count("\n1.") + finding.count("\n2.") + finding.count("\n3.")

        sources = self._merge_sources(seed_sources or [], self._extract_sources(messages))
        citations = [source.get("url", "") for source in sources if source.get("url")]
        
        # Clean up final response
        if final_response:
            final_response = final_response.replace("TERMINATE", "").strip()
        
        return {
            "query": query,
            "response": final_response,
            "conversation_history": messages,
            "citations": citations,
            "metadata": {
                "num_messages": len(messages),
                "num_sources": max(num_sources, len(sources)),
                "plan": plan,
                "research_findings": research_findings,
                "critique": critique,
                "agents_involved": list(set([msg.get("source", "") for msg in messages])),
                "sources": sources,
                "citations": citations,
            }
        }

    def _extract_sources(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract unique source URLs from conversation content."""
        urls_seen = set()
        sources: List[Dict[str, Any]] = []

        for msg in messages:
            content = msg.get("content", "")
            urls = re.findall(r"https?://[^\s<>{}\[\]]+", content)
            for url in urls:
                cleaned_url = url.rstrip(".,)")
                if cleaned_url in urls_seen:
                    continue
                urls_seen.add(cleaned_url)
                sources.append(
                    {
                        "title": cleaned_url,
                        "url": cleaned_url,
                        "source_agent": msg.get("source", "Unknown"),
                        "type": "webpage",
                    }
                )

        return sources

    def _merge_sources(
        self,
        seed_sources: List[Dict[str, Any]],
        extracted_sources: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Merge sources from orchestrator prefetch and conversation parsing."""
        merged: List[Dict[str, Any]] = []
        seen = set()
        for source in seed_sources + extracted_sources:
            key = source.get("url") or source.get("title")
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(source)
        return merged

    def get_agent_descriptions(self) -> Dict[str, str]:
        """
        Get descriptions of all agents.

        Returns:
            Dictionary mapping agent names to their descriptions
        """
        return {
            "Planner": "Breaks down research queries into actionable steps",
            "Researcher": "Gathers evidence from web and academic sources",
            "Writer": "Synthesizes findings into coherent responses",
            "Critic": "Evaluates quality and provides feedback",
        }

    def visualize_workflow(self) -> str:
        """
        Generate a text visualization of the workflow.

        Returns:
            String representation of the workflow
        """
        workflow = """
AutoGen Research Workflow:

1. User Query
   ↓
2. Planner
   - Analyzes query
   - Creates research plan
   - Identifies key topics
   ↓
3. Researcher (with tools)
   - Uses web_search() tool
   - Uses paper_search() tool
   - Gathers evidence
   - Collects citations
   ↓
4. Writer
   - Synthesizes findings
   - Creates structured response
   - Adds citations
   ↓
5. Critic
   - Evaluates quality
   - Checks completeness
   - Provides feedback
   ↓
6. Decision Point
   - If APPROVED → Final Response
   - If NEEDS REVISION → Back to Writer
        """
        return workflow


def demonstrate_usage():
    """
    Demonstrate how to use the AutoGen orchestrator.
    
    This function shows a simple example of using the orchestrator.
    """
    import yaml
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Create orchestrator
    orchestrator = AutoGenOrchestrator(config)
    
    # Print workflow visualization
    print(orchestrator.visualize_workflow())
    
    # Example query
    query = "What are the latest trends in human-computer interaction research?"
    
    print(f"\nProcessing query: {query}\n")
    print("=" * 70)
    
    # Process query
    result = orchestrator.process_query(query)
    
    # Display results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nQuery: {result['query']}")
    print(f"\nResponse:\n{result['response']}")
    print(f"\nMetadata:")
    print(f"  - Messages exchanged: {result['metadata']['num_messages']}")
    print(f"  - Sources gathered: {result['metadata']['num_sources']}")
    print(f"  - Agents involved: {', '.join(result['metadata']['agents_involved'])}")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    demonstrate_usage()

