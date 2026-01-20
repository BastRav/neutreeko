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
    //let _ = trainer.load("assets/models/3_100_MinMax6");

    trainer.train_opening(10);
    let _ = trainer.save("assets/models/1_10_opening");

    trainer.opponent = Some(MinMax::new(Color::Yellow, 4));
    trainer.training_loop(100);
    let _ = trainer.save("assets/models/2_100_MinMax4");

    trainer.opponent = Some(MinMax::new(Color::Yellow, 6));
    trainer.training_loop(100);
    let _ = trainer.save("assets/models/3_100_MinMax6");

    trainer.training_loop(100);
    let _ = trainer.save("assets/models/4_100_MinMax6");

    trainer.opponent = None;
    trainer.training_loop(100);
    let _ = trainer.save("assets/models/5_100_itself");

    trainer.train_opening(10);
    let _ = trainer.save("assets/models/6_10_opening");
    trainer.save_for_web();
}
