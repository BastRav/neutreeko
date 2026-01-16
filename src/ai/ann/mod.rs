mod block;
pub mod inputouput;

use crate::logic::{Color, Board, Direction};
use super::AI;
use log::info;

use inputouput::{board_to_input, output_to_moves};

use burn::{
    module::Module,
    nn::{
        BatchNorm, BatchNormConfig, PaddingConfig2d, Relu,
        conv::{Conv2d, Conv2dConfig},
    },
    tensor::{Device, Tensor, backend::Backend},
};

use block::{ResidualBlock, ValueHead, PolicyHead};

#[derive(Module, Debug)]
pub struct ANN<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B>,
    relu: Relu,
    layer1: ResidualBlock<B>,
    layer2: ResidualBlock<B>,
    layer3: ResidualBlock<B>,
    layer4: ResidualBlock<B>,
    value_head: ValueHead<B>,
    policy_head: PolicyHead<B>,
}

impl<B: Backend> ANN<B> {
    pub fn init(channels: usize, device: &Device<B>) -> Self {
        let conv1 = Conv2dConfig::new([2, channels], [7, 7])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(3, 3))
            .with_bias(false)
            .init(device);
        let bn1 = BatchNormConfig::new(channels).init(device);
        let relu = Relu::new();

        // Residual blocks
        let layer1 = ResidualBlock::new(channels, device);
        let layer2 = ResidualBlock::new(channels, device);
        let layer3 = ResidualBlock::new(channels, device);
        let layer4 = ResidualBlock::new(channels, device);

        let value_head = ValueHead::new(channels, device);
        let policy_head = PolicyHead::new(channels, device);

        Self {
            conv1,
            bn1,
            relu,
            layer1,
            layer2,
            layer3,
            layer4,
            value_head,
            policy_head,
        }
    }

    pub fn forward(&self, input: Tensor<B, 3>) -> (Tensor<B, 1>, Tensor<B, 1>) {
        let input_reshaped = input.reshape([1, 2, 5, 5]);
        //info!("ANN forward pass with input shape: {:?}", input_reshaped.shape());
        // First block
        let out = self.conv1.forward(input_reshaped);
        //info!("After conv1 shape: {:?}", out.shape());
        let out = self.bn1.forward(out);
        //info!("After bn1 shape: {:?}", out.shape());
        let out = self.relu.forward(out);
        //info!("After first block shape: {:?}", out.shape());

        // Residual blocks
        let out = self.layer1.forward(out);
        let out = self.layer2.forward(out);
        let out = self.layer3.forward(out);
        let out = self.layer4.forward(out);
        let out_copy = out.clone();
        //info!("After residual blocks shape: {:?}", out.shape());

        let value = self.value_head.forward(out);
        let policy = self.policy_head.forward(out_copy);
        (value, policy)
    }
    pub fn predict(&self, board:&Board) -> (f32, Vec<(f32, usize, Direction, Board)>) {
        let device = self.conv1.weight.device();
        let input = board_to_input(board, &device);
        let ann_output = self.forward(input);
        let board_eval: f32 = ann_output.0.to_data().into_vec().unwrap()[0];
        let moves_eval = output_to_moves(board, ann_output.1);
        (board_eval, moves_eval)
    }
}

#[derive(Clone)]
pub struct ANNSolo<B: Backend> {
    color: Color,
    ann: ANN<B>,
}

impl<B: Backend> AI for ANNSolo<B> {
    fn new(color: Color, _difficulty: usize) -> Self {
        let device = B::Device::default();
        Self {
            color,
            ann: ANN::init(32, &device),
        }
    }
    fn color(&self) -> &Color {
        &self.color
    }

    fn best_move(&mut self, board:&Board) -> (usize, Direction) {
        let (board_eval, moves_eval) = self.ann.predict(board);
        info!("ANN board evaluation for color {:?}: {}", self.color(), board_eval);
        let mut output = moves_eval;
        let output_print: Vec<(f32, usize, Direction)> = output.clone().into_iter().map(|x| (x.0, x.1, x.2)).collect();
        info!("ANN possible moves with probabilities: {:?}", output_print);
        let best_move = output.pop().unwrap();
        (best_move.1, best_move.2)
    }
}
