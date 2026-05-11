import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.LLMHandler import LLMHandler
from src.utils.SummaryHandler import SummaryHandler
from src.app.CampaignSummarizer import CampaignSummarizer

llm_handler = LLMHandler()
summary_handler = SummaryHandler(llm_handler)
CampaignSummarizer(llm_handler, summary_handler).run()
