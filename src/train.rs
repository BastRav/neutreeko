use neutreeko::ai::ann::train::ANNTrainer;
use burn::backend::{Autodiff, NdArray};

#[cfg(feature = "train")]
fn main() {
    let mut trainer: ANNTrainer<Autodiff<NdArray<f32>>> = ANNTrainer::new();
    trainer.training_loop();
}