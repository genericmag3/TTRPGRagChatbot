# Local D&D Campaign RAG Chatbot
## Summary
This project is an early implementation of a local RAG chatbot that will take dated notes as input, semantically chunk the notes and convert them into a vector database, and then pass relevant notes as context for queries about the campaign to the chatbot.
The chatbot will provide references to relevant note chunks at the end of each response. If there are no relevant notes found, the LLM is bypassed and a canned response is outputted. Custom animations have also been added for some extra flavor.

## Limitations and Future Updates
The prompt template is limited to my campaign, notes can only be inputted as a CSV file, notes cannot be re-uploaded, and the local LLM is pre-chosen.

Coming features:
- Modularize and generalize code to allow for easier future updates
- Allow for user to input custom instructions to LLM
- Automatically select LLM based on local system specs (GPU memory size)

## Demo
https://drive.google.com/file/d/1zYksJmQ10smPDopNKUE2VQ7tFtIlfYr_/view?usp=sharing

Hitch in vectorization animation is due to de-selecting chrome window during recording.
