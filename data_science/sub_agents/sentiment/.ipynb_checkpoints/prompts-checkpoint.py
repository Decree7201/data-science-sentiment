# prompts.py

SYSTEM_PROMPT = """
You are an AI agent designed to analyze user comments and assign a sentiment score.
Your task is to take a list of raw text comments and return a structured list
where each comment is paired with a score from 1 (very negative) to 5 (very positive).
"""

USER_PROMPT_TEMPLATE = """
Please analyze the following list of user comments and provide a score for each:
{comments_list}
"""

def format_output(scored_results):
    """
    Formats the final scored results into a readable string.
    """
    if not scored_results:
        return "No results to display."

    output = ["--- Comment Scoring Results ---"]
    for result in scored_results:
        output.append(f"Score: {result['score']}/5 | Comment: {result['text']}")
    return "\n".join(output)

