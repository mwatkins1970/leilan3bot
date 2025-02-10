import numpy as np
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from username_normalizer import UsernameNormalizer

# Configuration
RESULTS_PER_CATEGORY = {
    'gpt': 4,
    'opus': 4,
    'essay': 2,
    'interview': 2
}
SIMILARITY_METHOD = "max"  # or "mean"
DEFAULT_TEMPLATE = "rag_template.txt"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChunkMetadata:
    def __init__(self, label: str):
        self.label = label
        self.type, self.subtype = self._parse_label(label)
    
    @staticmethod
    def _parse_label(label: str) -> Tuple[str, str]:
        if not label or '_' not in label:
            return '', ''
        prefix, suffix = label.split('_', 1)
        if prefix == 'gpt3':
            return 'gpt', suffix
        elif prefix == 'opus':
            return 'opus', suffix
        return '', ''

class SubchunkData:
    def __init__(self, subchunks: List[str], embeddings: np.ndarray, parent_indices: List[int]):
        self.subchunks = subchunks
        self.embeddings = embeddings
        self.parent_indices = parent_indices  # Maps each subchunk to its parent chunk index

class ContextRetriever:
    def __init__(self, embeddings_dir="embeddings"):
        self.embeddings_dir = Path(embeddings_dir)
        self.subchunks_dir = self.embeddings_dir / "subchunked"
        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        self.load_data()

    async def retrieve_context_for_message(self, message_text: str) -> str:
        """
        Async method to retrieve context for a given message.
        Returns the formatted context string directly instead of writing to file.
        """
        # This method just calls the synchronous retrieve_contexts without await
        template = self.retrieve_contexts(message_text)
        return template

    @classmethod
    async def create(cls, embeddings_dir="embeddings"):
        """
        Async factory method to create and initialize a ContextRetriever instance.
        """
        retriever = cls(embeddings_dir)
        # Do any async initialization here if needed
        return retriever
    
    def load_data(self):
        """Load all necessary data files including subchunks."""
        try:
            logger.info("Loading data...")
            
            # Load dialogue data (used for both gpt and opus)
            self.dialogue_chunks = self.load_json(self.embeddings_dir / "dialogue_chunks_mpnet.json")
            dialogue_metadata_raw = self.load_json(self.embeddings_dir / "dialogue_metadata_mpnet.json")
            self.dialogue_metadata = [ChunkMetadata(label) for label in dialogue_metadata_raw]
            
            # Load dialogue subchunks
            self.dialogue_subchunks = SubchunkData(
                subchunks=self.load_json(self.subchunks_dir / "dialogue_texts_subchunked.json"),
                embeddings=np.load(self.embeddings_dir / "dialogue_embeddings_mpnet.npy"),
                parent_indices=self.load_json(self.subchunks_dir / "dialogue_metadata_subchunked.json")
            )
            
            # Create indices for gpt and opus chunks
            self.gpt_indices = [i for i, meta in enumerate(self.dialogue_metadata) if meta.type == 'gpt']
            self.opus_indices = [i for i, meta in enumerate(self.dialogue_metadata) if meta.type == 'opus']

            # Load essay data
            self.essay_chunks = self.load_json(self.embeddings_dir / "essay_chunks_mpnet.json")
            self.essay_metadata = self.load_json(self.embeddings_dir / "essay_metadata_mpnet.json")
            self.essay_subchunks = SubchunkData(
                subchunks=self.load_json(self.subchunks_dir / "essay_chunks_mpnet.json"),
                embeddings=np.load(self.embeddings_dir / "essay_embeddings_mpnet.npy"),
                parent_indices=self.load_json(self.subchunks_dir / "essay_metadata_mpnet.json")
            )

            # Load interview data
            self.interview_chunks = self.load_json(self.embeddings_dir / "interview_chunks_mpnet.json")
            self.interview_metadata = self.load_json(self.embeddings_dir / "interview_metadata_mpnet.json")
            self.interview_subchunks = SubchunkData(
                subchunks=self.load_json(self.subchunks_dir / "interview_chunks_mpnet.json"),
                embeddings=np.load(self.embeddings_dir / "interview_embeddings_mpnet.npy"),
                parent_indices=self.load_json(self.subchunks_dir / "interview_metadata_mpnet.json")
            )

            logger.info("Data loading complete")
            logger.info(f"Dialogue chunks: {len(self.dialogue_chunks)}")
            logger.info(f"Dialogue subchunks: {len(self.dialogue_subchunks.subchunks)}")
            logger.info(f"Essay chunks: {len(self.essay_chunks)}")
            logger.info(f"Essay subchunks: {len(self.essay_subchunks.subchunks)}")
            logger.info(f"Interview chunks: {len(self.interview_chunks)}")
            logger.info(f"Interview subchunks: {len(self.interview_subchunks.subchunks)}")

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def load_json(self, path: Path) -> Any:
        """Helper to load JSON files."""
        with open(path, 'r') as f:
            return json.load(f)

    def load_subchunk_data(self, chunks: List[str], base_name: str) -> SubchunkData:
        """Load subchunks, their embeddings, and parent indices for a document type."""
        print(f"\nLoading {base_name} subchunks...")
        
        # Handle different filename patterns for dialogue vs other types
        if base_name == "dialogue":
            subchunks = self.load_json(self.subchunks_dir / "dialogue_texts_subchunked.json")
            parent_indices = self.load_json(self.subchunks_dir / "dialogue_metadata_subchunked.json")
        else:
            subchunks = self.load_json(self.subchunks_dir / f"{base_name}_chunks_mpnet.json")
            parent_indices = self.load_json(self.subchunks_dir / f"{base_name}_metadata_mpnet.json")
        
        embeddings_path = self.embeddings_dir / f"{base_name}_embeddings_mpnet.npy"
        if not embeddings_path.exists():
            raise FileNotFoundError(f"Could not find embeddings file: {embeddings_path}")
        embeddings = np.load(embeddings_path)
        
        print(f"Loaded {len(subchunks)} subchunks with {len(parent_indices)} parent mappings")
        print(f"Embeddings shape: {embeddings.shape}")
        return SubchunkData(subchunks, embeddings, parent_indices)

    def get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for input text."""
        return self.model.encode(text, normalize_embeddings=True)

    def calculate_similarities(self, target_embedding: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
        """Calculate cosine similarities between target and embeddings."""
        print(f"\nCalculating similarities:")
        print(f"Target embedding shape: {target_embedding.shape}")
        print(f"Embeddings matrix shape: {embeddings.shape}")
        similarities = np.dot(embeddings, target_embedding)
        print(f"Resulting similarities shape: {similarities.shape}")
        print(f"Top 10 similarity scores: {sorted(similarities, reverse=True)[:10]}")
        return similarities

    def get_parent_index(self, parent_data: Union[Dict, int]) -> int:
        """Extract the parent index from various metadata formats."""
        if isinstance(parent_data, dict):
            if 'original_chunk_index' in parent_data:
                return parent_data['original_chunk_index']
            elif 'qa_index' in parent_data:
                return parent_data['qa_index']
            else:
                raise KeyError(f"Unrecognized parent index format: {parent_data}")
        return parent_data

    def get_chunk_similarities(self, subchunk_similarities: np.ndarray, 
                             parent_indices: List[Dict], 
                             method: str = "max") -> Dict[int, float]:
        """Calculate chunk similarities based on their subchunks."""
        chunk_similarities = defaultdict(list)
        
        # Group subchunk similarities by parent chunk
        print("\nGrouping subchunk similarities by parent chunk...")
        for subchunk_idx, similarity in enumerate(subchunk_similarities):
            try:
                parent_idx = self.get_parent_index(parent_indices[subchunk_idx])
                chunk_similarities[parent_idx].append(similarity)
                if subchunk_idx < 5:  #print first few for debugging
                    print(f"Subchunk {subchunk_idx} -> Parent {parent_idx} (similarity: {similarity:.3f})")
            except Exception as e:
                print(f"Error processing subchunk {subchunk_idx}: {str(e)}")
                print(f"Parent data: {parent_indices[subchunk_idx]}")
                continue
        
        # Calculate final similarity for each chunk based on method
        result = {}
        print(f"\nCalculating {method} similarities for {len(chunk_similarities)} chunks...")
        for chunk_idx, similarities in chunk_similarities.items():
            if method == "max":
                result[chunk_idx] = max(similarities)
            else:  # mean
                result[chunk_idx] = sum(similarities) / len(similarities)
            if len(result) < 5:  # print first few for debugging
                print(f"Chunk {chunk_idx}: {result[chunk_idx]:.3f} ({len(similarities)} subchunks)")
                
        return result

    def get_filtered_chunks(self, category: str, subchunk_data: SubchunkData, 
                          query_embedding: np.ndarray) -> List[Tuple[str, Dict, float, int]]:
        """Get chunks filtered by category and sorted by similarity."""
        print(f"\nProcessing {category.upper()} chunks:")
        
        # Calculate similarities with subchunks
        subchunk_similarities = self.calculate_similarities(query_embedding, subchunk_data.embeddings)
        
        # print top 50 most similar subchunks
        #print(f"\nTop 50 most similar {category} subchunks:")
        top_50_subchunk_indices = np.argsort(subchunk_similarities)[-50:][::-1]
        for i, idx in enumerate(top_50_subchunk_indices):
            sim = subchunk_similarities[idx]
            subchunk = subchunk_data.subchunks[idx]
            parent_idx = subchunk_data.parent_indices[idx]
            #print(f"\n{'-'*80}")
            #print(f"#{i+1} - Subchunk Index: {idx}")
            #print(f"Parent Chunk Index: {parent_idx}")
            #print(f"Similarity: {sim:.3f}")
            #print(f"Text: {subchunk}")
        
        # Get chunk similarities based on their subchunks
        chunk_similarities = self.get_chunk_similarities(
            subchunk_similarities, 
            subchunk_data.parent_indices, 
            SIMILARITY_METHOD
        )
        
        # Filter chunks based on category if needed
        if category in ['gpt', 'opus']:
            valid_indices = self.gpt_indices if category == 'gpt' else self.opus_indices
            chunk_similarities = {idx: sim for idx, sim in chunk_similarities.items() 
                               if idx in valid_indices}
        
        # Convert to sorted list of results
        results = []
        for chunk_idx, similarity in chunk_similarities.items():
            if category in ['gpt', 'opus']:
                chunk = self.dialogue_chunks[chunk_idx]
                metadata = self.dialogue_metadata[chunk_idx]
                meta_dict = {'label': metadata.label}
            else:
                chunk = (self.essay_chunks if category == 'essay' 
                        else self.interview_chunks)[chunk_idx]
                meta_dict = (self.essay_metadata if category == 'essay' 
                           else self.interview_metadata)[chunk_idx]
            
            results.append((chunk, meta_dict, similarity, chunk_idx))
        
        results.sort(key=lambda x: x[2], reverse=True)
        
        #print(f"\nTop 10 {category} chunks by similarity:")
        #for i, (chunk, meta, sim, idx) in enumerate(results[:10]):
            #print(f"#{i+1} - Index: {idx}, Similarity: {sim:.3f}")
            #if category in ['gpt', 'opus']:
                #print(f"Label: {meta['label']}")
            #print(f"First 100 chars: {chunk[:100]}...")
        
        return results

    def deduplicate_chunks(self, results: List[Tuple[str, Dict, float, int]], 
                          max_results: int = 10) -> List[Tuple[str, Dict, float, int]]:
        """Remove duplicate or highly similar chunks from results."""
        print("\nDeduplicating chunks:")
        #print(f"Initial number of chunks: {len(results)}")
        
        final_results = []
        considered_indices = set()
        checked_count = 0

        for result in results:
            text, meta, sim, index = result
            
            is_duplicate = any(text in other[0] or other[0] in text for other in final_results)
            if is_duplicate:
                print(f"Skipping duplicate chunk {index} (similarity: {sim:.3f})")
                continue
                
            final_results.append(result)
            considered_indices.add(index)
            checked_count += 1
            print(f"Added chunk {index} (similarity: {sim:.3f})")
            
            if len(final_results) >= max_results:
                print(f"Reached maximum results ({max_results})")
                break
                
            if checked_count > 100:
                print("WARNING: More than 100 chunks considered for deduplication.")
                break
        
        final_results = sorted(final_results, key=lambda x: x[2], reverse=True)
        print(f"Final number of chunks after deduplication: {len(final_results)}")
        return final_results

    def format_chunk(self, text: str, metadata: Dict, similarity: float, index: int) -> str:
            """Format a single chunk with metadata for output."""
            header = f"[segment index: {index}]\n[similarity: {similarity:.3f}]"
            
            # Add model/voice info if available
            if 'label' in metadata:
                chunk_meta = ChunkMetadata(metadata['label'])
                if chunk_meta.type == 'gpt':
                    header += f"\n[GPT-3 model: {chunk_meta.subtype}]"
                elif chunk_meta.type == 'opus':
                    header += f"\n[Opus voice: {chunk_meta.subtype}]"
            
            return f"{header}\n\n{text}\n\n{'-'*80}\n"

    def get_template(self) -> str:
        """Load or create the template string."""
        template_path = Path(DEFAULT_TEMPLATE)
        if template_path.exists():
            with open(template_path, 'r', encoding='utf-8') as f:
                return f.read()
        return "<gpt>\n\n<opus>\n\n<essay>\n\n<interview>"

    def retrieve_contexts(self, query: str) -> str:
        """Main method to retrieve and format relevant contexts."""
        normalizer = UsernameNormalizer()
        normalized_query = normalizer.normalize_message_history(query)
        print(f"\n{'='*80}\nProcessing query: {normalized_query}\n{'='*80}")
        
        query_embedding = self.get_embedding(normalized_query)
        print(f"\nGenerated query embedding with shape: {query_embedding.shape}")
        
        # Dictionary mapping categories to their subchunk data
        categories = {
            'gpt': self.dialogue_subchunks,
            'opus': self.dialogue_subchunks,
            'essay': self.essay_subchunks,
            'interview': self.interview_subchunks
        }

        # Process each category
        filled_sections = {}
        for category, subchunk_data in categories.items():
            print(f"\n{'='*40}\nProcessing {category.upper()} category\n{'='*40}")
            
            # Get chunks based on subchunk similarities
            results = self.get_filtered_chunks(category, subchunk_data, query_embedding)
            
            # Deduplicate and format results
            print(f"\nDeduplicating {category} results...")
            deduped_results = self.deduplicate_chunks(results, RESULTS_PER_CATEGORY[category])
            print(f"Final {category} results after deduplication: {len(deduped_results)}")
            
            filled_sections[category] = "".join([
                self.format_chunk(text, meta, sim, idx) 
                for text, meta, sim, idx in deduped_results
            ])

        # Fill template with results
        template = self.get_template()
        for tag, content in filled_sections.items():
            template = template.replace(f"<{tag}>", content)

        return template