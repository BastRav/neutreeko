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
    train();
    evaluate();
}

fn train() {
    let mut trainer: ANNTrainer<Autodiff<NdArray<f32>>, MinMax<NativePlatform>> = ANNTrainer::new();
    let result = trainer.load("assets/models/12_3_opening");
    if result.is_err() {println!("Could not load model");}

    // trainer.train_opening(3);
    // let _ = trainer.save("assets/models/7_3_opening");

    // trainer.opponent = Some(MinMax::new(Color::Yellow, 4));
    // trainer.training_loop(10);
    // let _ = trainer.save("assets/models/8_10_MinMax4");

    // trainer.training_loop(10);
    // let _ = trainer.save("assets/models/9_10_MinMax4");

    // trainer.training_loop(10);
    // let _ = trainer.save("assets/models/10_10_MinMax4");

    // trainer.opponent = None;
    trainer.training_loop(200);
    let _ = trainer.save("assets/models/13_200_itself");

    trainer.train_opening(3);
    let _ = trainer.save("assets/models/14_3_opening");
    trainer.save_for_web();
}

fn evaluate(){
    let mut trainer: ANNTrainer<Autodiff<NdArray<f32>>, MinMax<NativePlatform>> = ANNTrainer::new();
    trainer.opponent = Some(MinMax::new(Color::Yellow, 4));
    // let _ = trainer.load("assets/models/7_3_opening");
    // trainer.evaluate(2);
    // let _ = trainer.load("assets/models/8_10_MinMax4");
    // trainer.evaluate(2);
    // let _ = trainer.load("assets/models/9_10_MinMax4");
    // trainer.evaluate(2);
    // let _ = trainer.load("assets/models/10_10_MinMax4");
    // trainer.evaluate(2);
    let _ = trainer.load("assets/models/13_200_itself");
    trainer.evaluate(2);
    let _ = trainer.load("assets/models/14_3_opening");
    trainer.evaluate(2);
}
