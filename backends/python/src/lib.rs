mod logging;
mod management;

use nohash_hasher::BuildNoHashHasher;
use std::collections::HashMap;
use text_embeddings_backend_core::{
    Backend, BackendError, Batch, Embedding, Embeddings, ModelType, Pool, Predictions,
};

pub struct PythonBackend {
    backend_process: management::BackendProcess,
}

impl PythonBackend {
    pub fn new(
        model_path: String,
        dtype: String,
        model_type: ModelType,
        uds_path: String,
        otlp_endpoint: Option<String>,
    ) -> Result<Self, BackendError> {
        match model_type {
            ModelType::Classifier => {
                return Err(BackendError::Start(
                    "`classifier` model type is not supported".to_string(),
                ))
            }
            ModelType::Embedding(pool) => {
                if pool != Pool::Cls {
                    return Err(BackendError::Start(format!("{pool:?} is not supported")));
                }
                pool
            }
        };

        let backend_process =
            management::BackendProcess::new(model_path, dtype, uds_path, otlp_endpoint)?;

        Ok(Self { backend_process })
    }
}

impl Backend for PythonBackend {
    fn health(&self) -> Result<(), BackendError> {
        if self.backend_process.client.health().is_err() {
            return Err(BackendError::Unhealthy);
        }
        Ok(())
    }

    fn is_padded(&self) -> bool {
        false
    }

    fn embed(&self, batch: Batch) -> Result<Embeddings, BackendError> {
        if !batch.raw_indices.is_empty() {
            return Err(BackendError::Inference(
                "raw embeddings are not supported for the Python backend.".to_string(),
            ));
        }
        let batch_size = batch.len();

        let results = self
            .backend_process
            .client
            .embed(
                batch.input_ids,
                batch.token_type_ids,
                batch.position_ids,
                batch.cumulative_seq_lengths,
                batch.max_length,
            )
            .map_err(|err| BackendError::Inference(err.to_string()))?;
        let pooled_embeddings: Vec<Vec<f32>> = results.into_iter().map(|r| r.values).collect();

        let mut embeddings =
            HashMap::with_capacity_and_hasher(batch_size, BuildNoHashHasher::default());
        for (i, e) in pooled_embeddings.into_iter().enumerate() {
            embeddings.insert(i, Embedding::Pooled(e));
        }

        Ok(embeddings)
    }

    fn predict(&self, _batch: Batch) -> Result<Predictions, BackendError> {
        Err(BackendError::Inference(
            "`predict` is not implemented".to_string(),
        ))
    }
}
