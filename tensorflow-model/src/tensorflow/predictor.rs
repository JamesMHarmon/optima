use anyhow::Result;
use half::f16;
use itertools::Itertools;
use std::collections::HashMap;
use std::os::raw::c_int;
use std::path::Path;
use tensorflow::*;

pub struct Predictor {
    pub session: Session,
    pub input: OperationWithIndex,
    pub outputs: HashMap<String, OperationWithIndex>,
}

impl Predictor {
    pub fn new(path: &Path) -> Self {
        let mut graph = Graph::new();

        let model: SavedModelBundle =
            SavedModelBundle::load(&SessionOptions::new(), ["serve"], &mut graph, path)
                .expect("Expected to be able to load model");

        let signature = model
            .meta_graph_def()
            .get_signature(DEFAULT_SERVING_SIGNATURE_DEF_KEY)
            .unwrap_or_else(|_| {
                panic!(
                    "Failed to get signature: {}",
                    DEFAULT_SERVING_SIGNATURE_DEF_KEY
                )
            });

        let input = OperationWithIndex::new(
            signature
                .inputs()
                .iter()
                .next()
                .expect("Expected to find input"),
            &graph,
        );
        let outputs: HashMap<String, OperationWithIndex> = signature
            .outputs()
            .iter()
            .map(|signature| {
                (
                    signature.0.to_owned(),
                    OperationWithIndex::new(signature, &graph),
                )
            })
            .collect::<HashMap<_, _>>();
        let session = model.session;

        Self {
            session,
            input,
            outputs,
        }
    }

    pub(super) fn predict(&self, tensor: &Tensor<f16>) -> Result<AnalysisResults> {
        let mut session_run_args = SessionRunArgs::new();

        session_run_args.add_feed(&self.input.operation, self.input.index, tensor);

        let fetch_tokens = self
            .outputs
            .iter()
            .map(|(name, op)| {
                (
                    name.to_owned(),
                    session_run_args.request_fetch(&op.operation, op.index),
                    op.size,
                )
            })
            .collect_vec();

        self.session
            .run(&mut session_run_args)
            .expect("Expected to be able to run the model session");

        let outputs = fetch_tokens
            .into_iter()
            .map(|(name, fetch_token, size)| {
                (
                    name,
                    AnalysisResult {
                        tensor: session_run_args
                            .fetch(fetch_token)
                            .expect("Expected to be able to load output"),
                        size,
                    },
                )
            })
            .collect::<HashMap<String, AnalysisResult>>();

        Ok(AnalysisResults { outputs })
    }
}

pub(super) struct AnalysisResult {
    pub(super) tensor: Tensor<f16>,
    pub(super) size: usize,
}

pub(super) struct AnalysisResults {
    pub(super) outputs: HashMap<String, AnalysisResult>,
}

pub struct OperationWithIndex {
    pub name: String,
    pub operation: Operation,
    pub index: c_int,
    pub size: usize,
}

impl OperationWithIndex {
    pub(super) fn new(signature: (&String, &TensorInfo), graph: &Graph) -> Self {
        let (name, tensor_info) = signature;
        let shape: Option<Vec<Option<i64>>> = tensor_info.shape().clone().into();
        let size = shape
            .expect("Shape should be defined")
            .into_iter()
            .flatten()
            .product::<i64>() as usize;

        Self {
            name: name.to_owned(),
            operation: graph
                .operation_by_name_required(&tensor_info.name().name)
                .expect("Expected to find input operation"),
            index: tensor_info.name().index,
            size,
        }
    }
}
