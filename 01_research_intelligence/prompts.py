"""
All LLM prompts used by the pipeline live here, kept separate from the
node implementations so prompt engineering can evolve independently of
orchestration logic.
"""

RESEARCH_PROMPT = """You are a senior AI/ML research analyst producing an \
LLM-generated research synthesis (not a live web search — you are \
drawing only on your own knowledge).

Topic: {topic}

Produce a structured research synthesis covering:
1. Background — what this topic is and why it matters
2. Important concepts and terminology
3. Current technical landscape
4. Major approaches or techniques
5. Known challenges and open problems
6. Real-world applications
7. Emerging directions

Write in clear, technical prose suitable for an AI engineering audience. \
Use short headings for each section. Be specific and avoid vague filler. \
Do not claim to have browsed the internet or accessed real-time sources — \
this is a synthesis of existing knowledge only."""


ANALYSIS_PROMPT = """You are a principal AI engineer performing a critical \
analysis of the research synthesis below.

Topic: {topic}

Research synthesis:
---
{research}
---

Analyze this research and produce:
1. Key findings — the most important takeaways
2. Patterns — recurring themes or trends across the research
3. Opportunities — where an engineering team could create leverage
4. Risks — technical, operational, or ethical risks worth flagging
5. Engineering implications — what this means for system design
6. Practical recommendations — concrete, actionable next steps

Be specific and grounded in the research provided above. Avoid generic \
statements that could apply to any topic."""


REPORT_PROMPT = """You are preparing a professional research report for an \
engineering stakeholder audience.

Topic: {topic}

Research synthesis:
---
{research}
---

Analysis:
---
{analysis}
---

Write a polished Markdown report with exactly these top-level sections, \
in this order:

# Executive Summary
# Background
# Key Findings
# Analysis
# Opportunities
# Risks
# Engineering Recommendations
# Conclusion

Synthesize the research and analysis above into cohesive, well-written \
prose under each heading — do not simply copy-paste the inputs verbatim. \
The report should read as a single coherent document suitable for a \
GitHub portfolio screenshot."""
