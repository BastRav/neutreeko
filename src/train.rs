use neutreeko::{
    logic::Color,
    platform::NativePlatform,
    ai::{
        AI,
        ann::train::ANNTrainer,
        minmax::MinMax,
    }
};
use burn::backend::{Autodiff, NdArray};

#[cfg(feature = "train")]
fn main() {
    let mut trainer: ANNTrainer<Autodiff<NdArray<f32>>, MinMax<NativePlatform>> = ANNTrainer::new();
    let _ = trainer.load("assets/models/old");
    trainer.opponent = Some(MinMax::new(Color::Yellow, 6));
    trainer.training_loop(200);
    let _ = trainer.save("assets/models/new");
}
