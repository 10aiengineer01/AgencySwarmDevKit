from agency_swarm.tools import BaseTool
from pydantic import Field
import re
from openai import OpenAI
import os

class CheckRequirementsTool(BaseTool):
    """
    This tool analyzes project requirements to verify their completeness and feasibility.
    It identifies any missing or unclear elements that need further clarification.
    """

    information: str = Field(
        ..., description="The information from the user"
    )

    def run(self):
        """
        Analyzes the given requirements to identify missing or unclear elements.
        Returns a report highlighting areas that need further clarification.
        """
        
        api_key = os.getenv("OPENAI_API_KEY")

        current_info = self._shared_state.get("information", None)

        # Update Shared State
        if current_info is not None:
            updated_info = current_info + "\n" + self.information
        else:
            updated_info = self.information
        self._shared_state.set("information", updated_info)

        try:
            client = OpenAI(api_key=api_key)

            model_name = "gpt-4o"

            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": """
                                You are a system whose task is to determine whether the currently available information is sufficient...
                                INFORMATION: """ + current_info if current_info else ""
                            }
                        ]
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "New information: " + self.information
                            }
                        ]
                    },
                ],
                response_format={
                    "type": "text"
                },
                temperature=0,
                max_tokens=16383,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )

            if not response or not response.choices:
                return "No valid response received."

            return response.choices[0].message.content
        except Exception as e:
            return "An error occurred while checking requirements."