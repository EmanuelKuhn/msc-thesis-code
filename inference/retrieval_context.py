import datasets, faiss
import numpy as np

import typing

import torch

from typing import Protocol

class RetrievalContext(Protocol):
    def retrieve(self, query_ids,  top_k=50) -> np.ndarray:
        ...

class EmbeddingsRetrievalContext(RetrievalContext):
    def __init__(self, features: typing.Dict[str, torch.Tensor]):
        self.ids_all = features["id"].numpy()
        self.embeddings_all = (features["pred"] / features["pred"].norm(dim=1, keepdim=True)).numpy()    

        self.ds = datasets.Dataset.from_dict({"id": self.ids_all})

        self.ds.add_faiss_index_from_external_arrays(self.embeddings_all, "embedding", metric_type=faiss.METRIC_L2)

        to_rplan_id_lookup = {i: self.ds[i]["id"] for i in range(len(self.ds))}
        self.to_rplan_id = np.vectorize(lambda x: to_rplan_id_lookup[x], otypes=[np.integer])
    
    def retrieve(self, query_ids, query_embeddings=None, top_k=50) -> np.ndarray:

        if query_embeddings is None:

            # Get query embeddings by query ids, they should be in the same order as query_ids
            query_embeddings = np.concatenate([self.embeddings_all[np.argwhere(self.ids_all == query_id)[:, 0]] for query_id in query_ids], axis=0)

            assert len(query_embeddings) == len(query_ids), f"{len(query_embeddings)=}, {len(query_ids)=}"

        batch_results = self.ds.search_batch("embedding", query_embeddings, k=top_k+1)
        
        retrieval_ids_including_query = self.to_rplan_id(batch_results.total_indices)

        # Remove query from retrieval results
        filtered_retrievals = []

        for query_id, row in zip(query_ids, retrieval_ids_including_query):
            assert query_id in row, f"Query id not in retrieval results, {query_id=}, {row=}"
            
            filtered_retrievals.append(row[row != query_id])

        filtered_retrievals = np.array(filtered_retrievals)

        return filtered_retrievals
    

    def retrieve_by_embedding(self, query_embeddings, top_k=50) -> np.ndarray:

        batch_results = self.ds.search_batch("embedding", query_embeddings, k=top_k+1)
        
        return self.to_rplan_id(batch_results.total_indices)
