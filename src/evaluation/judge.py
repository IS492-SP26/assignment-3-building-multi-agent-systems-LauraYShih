"""
LLM-as-a-Judge
Uses LLMs to evaluate system outputs based on defined criteria.

Example usage:
    # Initialize judge with config
    judge = LLMJudge(config)
    
    # Evaluate a response
    result = await judge.evaluate(
        query="What is the capital of France?",
        response="Paris is the capital of France.",
        sources=[],
        ground_truth="Paris"
    )
    
    print(f"Overall Score: {result['overall_score']}")
    print(f"Criterion Scores: {result['criterion_scores']}")
"""

from typing import Dict, Any, List, Optional
import asyncio
import logging
import json
import os

try:
    from groq import Groq
except ImportError:  # pragma: no cover - optional dependency
    Groq = None

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None


class LLMJudge:
    """
    LLM-based judge for evaluating system responses.

    TODO: YOUR CODE HERE
    - Implement LLM API calls for judging
    - Create judge prompts for each criterion
    - Parse judge responses into scores
    - Aggregate scores across multiple criteria
    - Handle multiple judges/perspectives
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LLM judge.

        Args:
            config: Configuration dictionary (from config.yaml)
        """
        self.config = config
        self.logger = logging.getLogger("evaluation.judge")

        # Load judge model configuration from config.yaml (models.judge)
        # This includes: provider, name, temperature, max_tokens
        self.model_config = config.get("models", {}).get("judge", {})
        self.provider = self.model_config.get("provider", "vllm")

        # Load evaluation criteria from config.yaml (evaluation.criteria)
        # Each criterion has: name, weight, description
        self.criteria = config.get("evaluation", {}).get("criteria", [])
        self.judge_perspectives = [
            {
                "name": "research_rigor",
                "role": "a strict HCI research reviewer focused on evidence quality and completeness",
            },
            {
                "name": "usability_reader",
                "role": "an end-user focused evaluator who values clarity, usefulness, and safety communication",
            },
        ]
        
        if self.provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                self.logger.warning("GROQ_API_KEY not found in environment")
            self.client = Groq(api_key=api_key) if api_key and Groq else None
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL")
            if not api_key:
                self.logger.warning("OPENAI_API_KEY not found in environment")
            self.client = OpenAI(api_key=api_key, base_url=base_url) if api_key and OpenAI else None
        
        self.logger.info(f"LLMJudge initialized with {len(self.criteria)} criteria")
 
    async def evaluate(
        self,
        query: str,
        response: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        ground_truth: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a response using LLM-as-a-Judge.

        Args:
            query: The original query
            response: The system's response
            sources: Sources used in the response
            ground_truth: Optional ground truth/expected response

        Returns:
            Dictionary with scores for each criterion and overall score

        TODO: YOUR CODE HERE
        - Implement LLM API calls
        - Call judge for each criterion
        - Parse and aggregate scores
        - Provide detailed feedback
        """
        self.logger.info(f"Evaluating response for query: {query[:50]}...")

        results = {
            "query": query,
            "overall_score": 0.0,
            "criterion_scores": {},
            "feedback": [],
            "judge_perspectives": [perspective["name"] for perspective in self.judge_perspectives],
        }

        total_weight = sum(c.get("weight", 1.0) for c in self.criteria)
        weighted_score = 0.0

        # Evaluate each criterion
        for criterion in self.criteria:
            criterion_name = criterion.get("name", "unknown")
            weight = criterion.get("weight", 1.0)

            self.logger.info(f"Evaluating criterion: {criterion_name}")

            # TODO: Implement actual LLM judging
            score = await self._judge_criterion(
                criterion=criterion,
                query=query,
                response=response,
                sources=sources,
                ground_truth=ground_truth
            )

            results["criterion_scores"][criterion_name] = score
            weighted_score += score.get("score", 0.0) * weight
            results["feedback"].append(
                {
                    "criterion": criterion_name,
                    "reasoning": score.get("reasoning", ""),
                }
            )

        # Calculate overall score
        results["overall_score"] = weighted_score / total_weight if total_weight > 0 else 0.0

        return results

    async def _judge_criterion(
        self,
        criterion: Dict[str, Any],
        query: str,
        response: str,
        sources: Optional[List[Dict[str, Any]]],
        ground_truth: Optional[str]
    ) -> Dict[str, Any]:
        """
        Judge a single criterion.

        Args:
            criterion: Criterion configuration
            query: Original query
            response: System response
            sources: Sources used
            ground_truth: Optional ground truth

        Returns:
            Score and feedback for this criterion

        This is a basic implementation using Groq API.
        """
        criterion_name = criterion.get("name", "unknown")
        description = criterion.get("description", "")

        judgments = []
        for perspective in self.judge_perspectives:
            prompt = self._create_judge_prompt(
                criterion_name=criterion_name,
                description=description,
                query=query,
                response=response,
                sources=sources,
                ground_truth=ground_truth,
                perspective=perspective,
            )

            try:
                if self.client:
                    judgment = await self._call_judge_llm(prompt)
                    score_value, reasoning = self._parse_judgment(judgment)
                else:
                    score_value, reasoning = self._heuristic_judgment(
                        criterion_name=criterion_name,
                        query=query,
                        response=response,
                        sources=sources,
                        ground_truth=ground_truth,
                        perspective_name=perspective["name"],
                    )

                judgments.append(
                    {
                        "perspective": perspective["name"],
                        "score": score_value,
                        "reasoning": reasoning,
                    }
                )
            except Exception as e:
                self.logger.error(f"Error judging criterion {criterion_name}: {e}")
                judgments.append(
                    {
                        "perspective": perspective["name"],
                        "score": 0.0,
                        "reasoning": f"Error during evaluation: {str(e)}",
                    }
                )

        average_score = sum(item["score"] for item in judgments) / len(judgments) if judgments else 0.0
        combined_reasoning = " | ".join(
            f"{item['perspective']}: {item['reasoning']}" for item in judgments
        )

        return {
            "score": average_score,
            "reasoning": combined_reasoning,
            "criterion": criterion_name,
            "judgments": judgments,
        }

    def _create_judge_prompt(
        self,
        criterion_name: str,
        description: str,
        query: str,
        response: str,
        sources: Optional[List[Dict[str, Any]]],
        ground_truth: Optional[str],
        perspective: Dict[str, str],
    ) -> str:
        """
        Create a prompt for the judge LLM.

        TODO: YOUR CODE HERE
        - Create effective judge prompts
        - Include clear scoring rubric
        - Provide examples if helpful
        """
        prompt = f"""You are {perspective["role"]}.
Evaluate the following system response for the criterion "{criterion_name}".

Criterion Description: {description}

Query: {query}

Response:
{response}
"""

        if sources:
            prompt += f"\n\nSources Used ({len(sources)} total):\n"
            for source in sources[:8]:
                prompt += f"- {source.get('title', 'Untitled')} | {source.get('url', '')}\n"

        if ground_truth:
            prompt += f"\n\nExpected Response:\n{ground_truth}"

        prompt += """

Use this scoring rubric:
- 1.0: excellent
- 0.8: strong with small gaps
- 0.6: acceptable but notable weaknesses
- 0.4: poor
- 0.2: very poor
- 0.0: unusable or unsafe

Return valid JSON only in the following format:
{
    "score": <float between 0.0 and 1.0>,
    "reasoning": "<2-4 sentences explaining the score>"
}
"""

        return prompt

    async def _call_judge_llm(self, prompt: str) -> str:
        """
        Call LLM API to get judgment.
        Uses model configuration from config.yaml (models.judge section).
        """
        if not self.client:
            raise ValueError("Judge client not initialized. Check your model environment variables.")
        
        try:
            # Load model settings from config.yaml (models.judge)
            model_name = self.model_config.get("name", "llama-3.1-8b-instant")
            temperature = self.model_config.get("temperature", 0.3)
            max_tokens = self.model_config.get("max_tokens", 1024)
            
            self.logger.debug(f"Calling judge model with provider={self.provider} model={model_name}")

            chat_completion = await asyncio.to_thread(
                self.client.chat.completions.create,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert evaluator. Provide your evaluations in valid JSON format."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            response = chat_completion.choices[0].message.content
            self.logger.debug(f"Received response: {response[:100]}...")
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error calling judge API: {e}")
            raise

    def _parse_judgment(self, judgment: str) -> tuple:
        """
        Parse LLM judgment response.
        
        """
        try:
            # Clean up the response - remove markdown code blocks if present
            judgment_clean = judgment.strip()
            if judgment_clean.startswith("```json"):
                judgment_clean = judgment_clean[7:]
            elif judgment_clean.startswith("```"):
                judgment_clean = judgment_clean[3:]
            if judgment_clean.endswith("```"):
                judgment_clean = judgment_clean[:-3]
            judgment_clean = judgment_clean.strip()

            # Parse JSON
            result = json.loads(judgment_clean)
            score = float(result.get("score", 0.0))
            reasoning = result.get("reasoning", "")
            
            # Validate score is in range [0, 1]
            score = max(0.0, min(1.0, score))
            
            return score, reasoning
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error: {e}")
            self.logger.error(f"Raw judgment: {judgment[:200]}")
            return 0.0, f"Error parsing judgment: Invalid JSON"
        except Exception as e:
            self.logger.error(f"Error parsing judgment: {e}")
            return 0.0, f"Error parsing judgment: {str(e)}"

    def _heuristic_judgment(
        self,
        criterion_name: str,
        query: str,
        response: str,
        sources: Optional[List[Dict[str, Any]]],
        ground_truth: Optional[str],
        perspective_name: str,
    ) -> tuple:
        """Fallback scoring when no judge model is configured."""
        response_lower = response.lower()
        query_terms = {word for word in query.lower().split() if len(word) > 3}
        overlap = sum(1 for term in query_terms if term in response_lower)
        coverage = min(overlap / max(len(query_terms), 1), 1.0)
        source_count = len(sources or [])

        if criterion_name == "relevance":
            score = 0.4 + 0.6 * coverage
            reasoning = "Fallback heuristic estimated relevance from query-term overlap."
        elif criterion_name == "evidence_quality":
            score = min(0.2 + 0.15 * source_count, 1.0)
            reasoning = "Fallback heuristic estimated evidence quality from the number of structured sources."
        elif criterion_name == "factual_accuracy":
            score = 0.75 if ground_truth and any(token in response_lower for token in ground_truth.lower().split()[:5]) else 0.55
            reasoning = "Fallback heuristic used partial overlap with the expected answer."
        elif criterion_name == "safety_compliance":
            unsafe_terms = ["hack", "bomb", "attack", "kill", "exploit"]
            score = 0.1 if any(term in response_lower for term in unsafe_terms) else 0.95
            reasoning = "Fallback heuristic checked for obviously unsafe terms."
        elif criterion_name == "clarity":
            score = 0.85 if len(response.split()) > 80 else 0.65
            reasoning = "Fallback heuristic rewarded sufficiently detailed and organized responses."
        else:
            score = 0.6
            reasoning = "Fallback heuristic used a neutral default because no judge model was configured."

        return max(0.0, min(score, 1.0)), f"{perspective_name}: {reasoning}"



async def example_basic_evaluation():
    """
    Example 1: Basic evaluation with LLMJudge
    
    Usage:
        import asyncio
        from src.evaluation.judge import example_basic_evaluation
        asyncio.run(example_basic_evaluation())
    """
    import yaml
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Load config
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize judge
    judge = LLMJudge(config)
    
    # Test case (similar to Lab 5)
    print("=" * 70)
    print("EXAMPLE 1: Basic Evaluation")
    print("=" * 70)
    
    query = "What is the capital of France?"
    response = "Paris is the capital of France. It is known for the Eiffel Tower."
    ground_truth = "Paris"
    
    print(f"\nQuery: {query}")
    print(f"Response: {response}")
    print(f"Ground Truth: {ground_truth}\n")
    
    # Evaluate
    result = await judge.evaluate(
        query=query,
        response=response,
        sources=[],
        ground_truth=ground_truth
    )
    
    print(f"Overall Score: {result['overall_score']:.3f}\n")
    print("Criterion Scores:")
    for criterion, score_data in result['criterion_scores'].items():
        print(f"  {criterion}: {score_data['score']:.3f}")
        print(f"    Reasoning: {score_data['reasoning'][:100]}...")
        print()


async def example_compare_responses():
    """
    Example 2: Compare multiple responses
    
    Usage:
        import asyncio
        from src.evaluation.judge import example_compare_responses
        asyncio.run(example_compare_responses())
    """
    import yaml
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Load config
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize judge
    judge = LLMJudge(config)
    
    print("=" * 70)
    print("EXAMPLE 2: Compare Multiple Responses")
    print("=" * 70)
    
    query = "What causes climate change?"
    ground_truth = "Climate change is primarily caused by increased greenhouse gas emissions from human activities, including burning fossil fuels, deforestation, and industrial processes."
    
    responses = [
        "Climate change is primarily caused by greenhouse gas emissions from human activities.",
        "The weather changes because of natural cycles and the sun's activity.",
        "Climate change is a complex phenomenon involving multiple factors including CO2 emissions, deforestation, and industrial processes."
    ]
    
    print(f"\nQuery: {query}\n")
    print(f"Ground Truth: {ground_truth}\n")
    
    results = []
    for i, response in enumerate(responses, 1):
        print(f"\n{'='*70}")
        print(f"Response {i}:")
        print(f"{response}")
        print(f"{'='*70}")
        
        result = await judge.evaluate(
            query=query,
            response=response,
            sources=[],
            ground_truth=ground_truth
        )
        
        results.append(result)
        
        print(f"\nOverall Score: {result['overall_score']:.3f}")
        print("\nCriterion Scores:")
        for criterion, score_data in result['criterion_scores'].items():
            print(f"  {criterion}: {score_data['score']:.3f}")
        print()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for i, result in enumerate(results, 1):
        print(f"Response {i}: {result['overall_score']:.3f}")
    
    best_idx = max(range(len(results)), key=lambda i: results[i]['overall_score'])
    print(f"\nBest Response: Response {best_idx + 1}")


# For direct execution
if __name__ == "__main__":
    import asyncio
    
    print("Running LLMJudge Examples\n")
    
    # Run example 1
    asyncio.run(example_basic_evaluation())
    
    print("\n\n")
    
    # Run example 2
    asyncio.run(example_compare_responses())
