from ProjectPlanningAgency.ProjectPlannerAgent import ProjectPlannerAgent
from agency_swarm import Agency, set_openai_key
import os

class create_initiall_prompt:

    def __init__(self, key: str) -> None:
        set_openai_key(key)
        self.project_planner_agent = ProjectPlannerAgent()

        self.agency = Agency([self.project_planner_agent],
                        shared_instructions='./agency_manifesto.md',  # shared instructions for all agents
                        )

    def run(self, prompt: str) -> str:
        return self.agency.get_completion(prompt)