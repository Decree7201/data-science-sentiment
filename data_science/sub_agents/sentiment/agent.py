# agent.py

from .tools import create_scoring_model, score_comments_tool
from .prompts import SYSTEM_PROMPT, format_output

class CommentScoringAgent:
    """
    An agent that analyzes user comments to assign a sentiment score.
    """
    def __init__(self):
        """
        Initializes the agent by loading the necessary NLP model.
        """
        print("Initializing Comment Scoring Agent...")
        print(SYSTEM_PROMPT)
        self.tokenizer, self.model = create_scoring_model()

    def run(self, comments_to_score):
        """
        Runs the sentiment analysis on a list of comments.

        Args:
            comments_to_score (list of str): The list of comments to analyze.

        Returns:
            A formatted string containing the scored results.
        """
        if not self.model or not self.tokenizer:
            return "Agent could not be initialized. Cannot run."

        if not comments_to_score:
            return "No comments provided to score."

        # Use the dedicated tool to get the scores
        scored_results = score_comments_tool(comments_to_score, self.tokenizer, self.model)

        # Format the output using a template from prompts.py
        formatted_results = format_output(scored_results)
        
        return formatted_results

# --- Main execution block ---
if __name__ == "__main__":
    # This is where you would provide the 200 comments from your database.
    # For this example, we'll use a small sample.
    sample_comments_from_db = [
        "This is an amazing product, I absolutely love it!",
        "It's okay, but I've seen better. It does the job.",
        "I'm really disappointed with the quality. It broke after one day.",
        "A decent effort, but it's lacking some key features.",
        "Fantastic customer service and a brilliant piece of kit.",
        "Worst purchase I have ever made. Do not recommend.",
    ]
    
    # 1. Create an instance of the agent
    agent = CommentScoringAgent()

    # 2. Run the agent with your data
    final_output = agent.run(sample_comments_from_db)

    # 3. Print the final, formatted output
    print("\n" + final_output)

