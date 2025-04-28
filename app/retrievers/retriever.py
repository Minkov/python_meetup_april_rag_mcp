class Retriever:
    def __init__(self, vector_db):
        self.vector_db = vector_db
        
    def retrieve(self, query, top_k=5):
        """Retrieve relevant documents based on a query"""
        results = self.vector_db.search(query, top_k=top_k)
        return results
        
    def format_for_context(self, results):
        """Format search results for use in LLM context"""
        if not results:
            return "No relevant information found."
            
        context_parts = []
        
        for i, result in enumerate(results, start=1):
            context_parts.append(f"Document {i}:")
            context_parts.append(f"Relevance: {result['similarity']:.4f}")
            
            # Add metadata if available
            if result["metadata"]:
                context_parts.append("Metadata:")
                for key, value in result["metadata"].items():
                    context_parts.append(f"- {key}: {value}")
                    
            # Add document content
            context_parts.append("Content:")
            context_parts.append(result["text"])
            context_parts.append("-" * 40)  # Separator
            
        return "\n".join(context_parts)