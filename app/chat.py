import os
from app.hr_system import HRIntelligenceSystem


class Chat:
    def __init__(self):
        self.hr_system = HRIntelligenceSystem(os.getenv("OPENAI_API_KEY"))
        self.conversation_history = []
        self.last_retrieved_docs = []
        
    def add_message(self, role: str, content: str):
        self.conversation_history.append({"role": role, "content": content})
        
    def get_conversation_context(self, max_messages: int = 5) -> str:
        """Get recent conversation history as formatted context."""
        recent_messages = self.conversation_history[-max_messages:] if len(self.conversation_history) > max_messages else self.conversation_history
        formatted = []
        for msg in recent_messages:
            formatted.append(f"{msg['role'].upper()}: {msg['content']}")
        return "\n\n".join(formatted)
        
    def ask(self, query: str):
        self.add_message("user", query)
        
        conversation_context = self.get_conversation_context()
        
        result = self.hr_system.ask(query, context=conversation_context)
        
        self.last_retrieved_docs = result["retrieved_docs"]
        
        self.add_message("assistant", result["response"])
        
        return result