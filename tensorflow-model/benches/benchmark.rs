use std::path::Path;

use arimaa::{GameState, Mapper};
use criterion::{Criterion, criterion_group, criterion_main};
use rayon::iter::{ParallelBridge, ParallelIterator};
use tensorflow::{SessionRunArgs, Tensor};
use tensorflow_model::{InputMap, Mode, Predictor};

criterion_group!(benches, bench_tensorflow_infer);
criterion_main!(benches);

/// FINDINGS:
/// * Having a static batch size is better than a random batch size. I.E. a consistant 4096 is 43% faster than a 4025 then a 4096 then a 4012, etc.
/// * Reading and indexing into the output tensor is as performant as a normal slice.
fn bench_tensorflow_infer(c: &mut Criterion) {
    let game_state: &GameState = &"
            1g
            +-----------------+
            8| h c d m r d c h |
            7|   r r r e r r r |
            6|     x     x     |
            5|                 |
            4| r C             |
            3|     x     x     |
            2| R R R R E R R R |
            1| H   D M R D C H |
            +-----------------+
                a b c d e f g h
            "
    .parse()
    .unwrap();

    // let valid_actions = game_state.valid_actions();

    let predictor = Predictor::new(Path::new(
        "/home/james/arimaa-client/Arimaa_runs/run-1/5b64f_00100",
    ));

    let batch_size = 4096;
    let dimensions = [8, 8, 18];
    let input_len = dimensions.iter().product::<u64>() as usize;

    let mapper = Mapper::new();

    c.bench_function("bench infer", |b| {
        b.iter(|| {
            // let mut rng = rand::thread_rng();
            // let batch_size = rng.gen_range((batch_size - 1000)..=batch_size);

            let mut input: Tensor<half::f16> = Tensor::new(&[
                batch_size as u64,
                dimensions[0],
                dimensions[1],
                dimensions[2],
            ]);

            std::iter::repeat_with(|| game_state)
                .take(batch_size)
                .zip(input.chunks_mut(input_len))
                .par_bridge()
                .for_each(|(state_to_analyse, tensor_chunk)| {
                    mapper.game_state_to_input(state_to_analyse, tensor_chunk, Mode::Infer);
                });

            let mut output_step = SessionRunArgs::new();
            output_step.add_feed(&predictor.input.operation, predictor.input.index, &input);
            let value_head_fetch_token = output_step.request_fetch(
                &predictor.outputs["value_head"].operation,
                predictor.outputs["value_head"].index,
            );
            let policy_head_fetch_token = output_step.request_fetch(
                &predictor.outputs["policy_head"].operation,
                predictor.outputs["policy_head"].index,
            );
            let moves_left_head_fetch_token = output_step.request_fetch(
                &predictor.outputs["moves_left_head"].operation,
                predictor.outputs["moves_left_head"].index,
            );

            predictor
                .session
                .run(&mut output_step)
                .expect("Expected to be able to run the model session");

            let value_head_output: Tensor<half::f16> = output_step
                .fetch(value_head_fetch_token)
                .expect("Expected to be able to load value_head output");
            let policy_head_output: Tensor<half::f16> = output_step
                .fetch(policy_head_fetch_token)
                .expect("Expected to be able to load policy_head output");
            let moves_left_head_output: Tensor<half::f16> = output_step
                .fetch(moves_left_head_fetch_token)
                .expect("Expected to be able to load moves_left_head output");

            let mut value_head_output_vec: Vec<half::f16> =
                Vec::with_capacity(value_head_output.len());

            value_head_output_vec.resize(value_head_output.len(), half::f16::ZERO);
            value_head_output_vec.copy_from_slice(&value_head_output);

            let mut policy_head_output_vec: Vec<half::f16> =
                Vec::with_capacity(policy_head_output.len());

            policy_head_output_vec.resize(policy_head_output.len(), half::f16::ZERO);
            policy_head_output_vec.copy_from_slice(&policy_head_output);

            let mut moves_left_head_output_vec: Vec<half::f16> =
                Vec::with_capacity(moves_left_head_output.len());

            moves_left_head_output_vec.resize(moves_left_head_output.len(), half::f16::ZERO);
            moves_left_head_output_vec.copy_from_slice(&moves_left_head_output);

            let policy_head_output_len = policy_head_output_vec.len() / batch_size;
            for i in 0..batch_size {
                criterion::black_box(mapper.policy_to_valid_actions(
                    game_state,
                    &policy_head_output
                        [(i * policy_head_output_len)..((i + 1) * policy_head_output_len)],
                ));
            }

            criterion::black_box(value_head_output_vec);
            criterion::black_box(policy_head_output_vec);
            criterion::black_box(moves_left_head_output_vec);
        });
    });
}
