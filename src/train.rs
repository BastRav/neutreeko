use neutreeko::ai::ann::train::ANNTrainer;
use burn::backend::{Autodiff, NdArray};

#[cfg(feature = "train")]
fn main() {
    let mut trainer: ANNTrainer<Autodiff<NdArray<f32>>> = ANNTrainer::new();
    //let _ = trainer.load("assets/models/old");
    trainer.training_loop(2);
    let _ = trainer.save("assets/models/new");
}