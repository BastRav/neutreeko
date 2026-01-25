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
    // train();
    evaluate();
}

fn train() {
    let mut trainer: ANNTrainer<Autodiff<NdArray<f32>>, MinMax<NativePlatform>> = ANNTrainer::new();
    // let _ = trainer.load("assets/models/3_100_MinMax6");

    trainer.train_opening(3);
    let _ = trainer.save("assets/models/1_3_opening");

    trainer.opponent = Some(MinMax::new(Color::Yellow, 4));
    trainer.training_loop(10);
    let _ = trainer.save("assets/models/2_10_MinMax4");

    trainer.training_loop(10);
    let _ = trainer.save("assets/models/3_10_MinMax4");

    trainer.training_loop(10);
    let _ = trainer.save("assets/models/4_10_MinMax4");

    trainer.opponent = None;
    trainer.training_loop(10);
    let _ = trainer.save("assets/models/5_10_itself");

    trainer.train_opening(1);
    let _ = trainer.save("assets/models/6_1_opening");
    trainer.save_for_web();
}

fn evaluate(){
    let mut trainer: ANNTrainer<Autodiff<NdArray<f32>>, MinMax<NativePlatform>> = ANNTrainer::new();
    trainer.opponent = Some(MinMax::new(Color::Yellow, 4));
    let _ = trainer.load("assets/models/1_3_opening");
    trainer.evaluate(2);
    let _ = trainer.load("assets/models/2_10_MinMax4");
    trainer.evaluate(2);
    let _ = trainer.load("assets/models/3_10_MinMax4");
    trainer.evaluate(2);
    let _ = trainer.load("assets/models/4_10_MinMax4");
    trainer.evaluate(2);
    let _ = trainer.load("assets/models/5_10_itself");
    trainer.evaluate(2);
    let _ = trainer.load("assets/models/6_1_opening");
    trainer.evaluate(2);
}
