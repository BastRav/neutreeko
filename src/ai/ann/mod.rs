mod block;
mod utils;
#[cfg(feature = "train")]
pub mod train;

use std::marker::PhantomData;

use crate::{logic::{Board, Color, Direction}, platform::Platform};
use super::AI;

use utils::{board_to_input, output_to_moves};

use burn::{
    module::Module,
    nn::{
        BatchNorm, BatchNormConfig, PaddingConfig2d, Relu,
        conv::{Conv2d, Conv2dConfig},
    },
    tensor::{Device, Tensor, backend::Backend},
    config::Config,
};
use burn_store::{ModuleSnapshot, BurnpackStore};

use block::{ResidualBlock, ValueHead, PolicyHead};

struct PolicyValueOutput<B: Backend> {
    value: Tensor<B, 2>,
    policy: Tensor<B, 4>,
}

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
    fn forward(&self, input: Tensor<B, 4>) -> PolicyValueOutput<B> {
        // Input shape: [1, 2, 5, 5]

        // Subsequent blocks assume 32 channels
        
        // First block
        let out = self.conv1.forward(input); // [1, 32, 5, 5]
        //info!("After conv1 shape: {:?}", out.shape());
        let out = self.bn1.forward(out); // [1, 32, 5, 5]
        //info!("After bn1 shape: {:?}", out.shape());
        let out = self.relu.forward(out); // [1, 32, 5, 5]
        //info!("After first block shape: {:?}", out.shape());

        // Residual blocks
        let out = self.layer1.forward(out); // [1, 32, 5, 5]
        let out = self.layer2.forward(out); // [1, 32, 5, 5]
        let out = self.layer3.forward(out); // [1, 32, 5, 5]
        let out = self.layer4.forward(out); // [1, 32, 5, 5]
        let out_copy = out.clone();
        //info!("After residual blocks shape: {:?}", out.shape());

        let value = self.value_head.forward(out); // [1, 1]
        let policy = self.policy_head.forward(out_copy); // [1, 8, 5, 5]
        PolicyValueOutput { value, policy }
    }

    pub fn predict(&self, board:&Board) -> (f32, Vec<(f32, usize, Direction, Board)>) {
        let device = self.conv1.weight.device();
        let input = board_to_input(board, &device);
        let ann_output = self.forward(input);
        let board_eval: f32 = ann_output.value.to_data().into_vec().unwrap()[0];
        let moves_eval = output_to_moves(board, ann_output.policy);
        (board_eval, moves_eval)
    }
}

#[derive(Config, Debug)]
pub struct ANNConfig {
    channels: usize,
}

impl ANNConfig {
    pub fn init<B: Backend>(channels: usize, device: &Device<B>) -> ANN<B> {
        let conv1 = Conv2dConfig::new([2, channels], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Same)
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

        ANN {
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

    pub fn init_from_data<B: Backend>(channels: usize, device: &Device<B>) -> ANN<B> {
        let mut ann = ANNConfig::init(channels, device);
        static DATA: &[u8] = include_bytes!("../../../assets/models/web/model.bpk");
        let mut store = BurnpackStore::from_static(DATA);
        let _ = ann.load_from(&mut store);
        ann
    }
}


#[derive(Clone)]
pub struct ANNSolo<B: Backend, O:Platform> {
    color: Color,
    ann: ANN<B>,
    _platform: PhantomData<O>,
}

impl<B: Backend, O: Platform> AI<O> for ANNSolo<B, O> {
    fn new(color: Color, _difficulty: usize) -> Self {
        let device = B::Device::default();
        Self {
            color,
            ann: ANNConfig::init_from_data(32, &device),
            _platform: PhantomData,
        }
    }

    fn color(&self) -> &Color {
        &self.color
    }

    fn set_color(&mut self, color:Color){
        self.color = color;
    }

    fn give_all_options(&mut self, board:&Board, verbose: bool) -> (f32, Vec<(f32, usize, Direction)>) {
        let (board_eval, moves_eval) = self.ann.predict(board);
        if verbose {
            O::print(&format!("ANN board evaluation for color {:?}: {}", self.color(), board_eval));
        }
        (board_eval, moves_eval.into_iter().map(|x| (x.0, x.1, x.2)).collect())
    }
}
