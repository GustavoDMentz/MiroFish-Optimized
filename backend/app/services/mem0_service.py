import os
from mem0 import Memory
from ..config import Config
from ..utils.logger import get_logger

logger = get_logger('mirofish.mem0_service')

class Mem0Service:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Mem0Service, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
        
    def __init__(self):
        if self._initialized:
            return
            
        try:
            # Configurar o ChromaDB local no diretório de uploads
            chroma_db_path = os.path.join(Config.UPLOAD_FOLDER, "mem0_chromadb")
            os.makedirs(chroma_db_path, exist_ok=True)
            
            # Chaves e Base URL das configurações do MiroFish
            embedding_model = Config.EMBEDDING_MODEL_NAME or "text-embedding-3-small"
            embedding_api_key = Config.EMBEDDING_API_KEY or Config.LLM_API_KEY
            embedding_base_url = Config.EMBEDDING_BASE_URL or Config.LLM_BASE_URL
            
            # Fallback para OpenAI se chaves gerais da LLM estiverem presentes
            llm_model = Config.LLM_MODEL_NAME or "gpt-4o-mini"
            llm_api_key = Config.LLM_API_KEY
            llm_base_url = Config.LLM_BASE_URL

            config = {
                "vector_store": {
                    "provider": "chroma",
                    "config": {
                        "path": chroma_db_path,
                        "collection_name": "mirofish_agent_memory",
                    }
                },
                "embedder": {
                    "provider": "openai",
                    "config": {
                        "model": embedding_model,
                        "api_key": embedding_api_key,
                        "openai_base_url": embedding_base_url,
                    }
                },
                "llm": {
                    "provider": "openai",
                    "config": {
                        "model": llm_model,
                        "api_key": llm_api_key,
                        "openai_base_url": llm_base_url,
                    }
                }
            }
            
            logger.info(f"Inicializando Mem0 com Embedder model={embedding_model}, base_url={embedding_base_url}")
            self.memory = Memory.from_config(config)
            self._initialized = True
            logger.info("Mem0 Service inicializado com sucesso usando Chroma local e embeddings online.")
        except Exception as e:
            self.memory = None
            logger.error(f"Erro ao inicializar o serviço Mem0: {e}")
            
    def add_memory(self, text: str, user_id: str, metadata: dict = None):
        """Adiciona memória para um agente específico"""
        if not self.memory:
            logger.warning("Serviço Mem0 não inicializado. Ignorando add_memory.")
            return None
        try:
            metadata = metadata or {}
            return self.memory.add(text, user_id=user_id, metadata=metadata)
        except Exception as e:
            logger.error(f"Erro ao adicionar memória no Mem0 para {user_id}: {e}")
            return None
            
    def search_memory(self, query: str, user_id: str = None, limit: int = 10) -> list:
        """Busca memórias para um agente ou de forma global"""
        if not self.memory:
            logger.warning("Serviço Mem0 não inicializado. Retornando busca vazia.")
            return []
        try:
            filters = {"user_id": user_id} if user_id else None
            return self.memory.search(query, filters=filters, limit=limit)
        except Exception as e:
            logger.error(f"Erro ao buscar memória no Mem0 para {user_id}: {e}")
            return []

            
    def get_all_memories(self, user_id: str = None) -> list:
        """Obtém todas as memórias salvas"""
        if not self.memory:
            return []
        try:
            return self.memory.get_all(user_id=user_id)
        except Exception as e:
            logger.error(f"Erro ao obter memórias no Mem0 para {user_id}: {e}")
            return []
            
    def delete_memory(self, memory_id: str):
        """Deleta uma memória específica por ID"""
        if not self.memory:
            return
        try:
            self.memory.delete(memory_id)
        except Exception as e:
            logger.error(f"Erro ao deletar memória {memory_id} no Mem0: {e}")
