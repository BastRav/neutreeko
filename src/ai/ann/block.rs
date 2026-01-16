use burn::{
    module::Module,
    nn::{
        BatchNorm, BatchNormConfig, PaddingConfig2d, Relu, Tanh, Linear, LinearConfig,
        conv::{Conv2d, Conv2dConfig},
    },
    tensor::{Device, Tensor, backend::Backend, activation::{softmax, sigmoid}},
};
//use log::info;

/// ResNet [basic residual block](https://paperswithcode.com/method/residual-block) implementation.
#[derive(Module, Debug)]
pub struct ResidualBlock<B: Backend> {
  conv1: Conv2d<B>,
  bn1: BatchNorm<B>,
  relu: Relu,
  conv2: Conv2d<B>,
  bn2: BatchNorm<B>,
}

impl<B: Backend> ResidualBlock<B> {
    pub fn new(channels: usize, device: &Device<B>) -> Self {
        // conv3x3
        let conv1 = Conv2dConfig::new([channels, channels], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Same)
            .with_bias(false)
            .init(device);
        let bn1 = BatchNormConfig::new(channels).init(device);
        let relu = Relu::new();
        // conv3x3
        let conv2 = Conv2dConfig::new([channels, channels], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Same)
            .with_bias(false)
            .init(device);
        let bn2 = BatchNormConfig::new(channels).init(device);

        Self {
            conv1,
            bn1,
            relu,
            conv2,
            bn2,
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        //info!("Residual block forward pass with input shape: {:?}", input.shape());
        let identity = input.clone();

        // Conv block
        let out = self.conv1.forward(input);
        let out = self.bn1.forward(out);
        let out = self.relu.forward(out);
        let out = self.conv2.forward(out);
        let out = self.bn2.forward(out);

        // Skip connection
        let out = out + identity;

        // Activation
        let out = self.relu.forward(out);
        //info!("Residual block output shape: {:?}", out.shape());
        out
    }
}

#[derive(Module, Debug)]
pub struct ValueHead<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B>,
    relu: Relu,
    tanh: Tanh,
    linear: Linear<B>,
}

impl<B: Backend> ValueHead<B> {
    pub fn new(channels: usize, device: &Device<B>) -> Self {
        // conv1x1
        let conv1 = Conv2dConfig::new([channels, channels], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Same)
            .with_bias(false)
            .init(device);
        let bn1 = BatchNormConfig::new(channels).init(device);
        let relu = Relu::new();
        let tanh = Tanh::new();
        let linear = LinearConfig::new(channels * 3 * 3, 1).init(device);

        Self {
            conv1,
            bn1,
            relu,
            tanh,
            linear,
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 1> {
        //info!("Value head forward pass with input shape: {:?}", input.shape());
        let out = self.conv1.forward(input);
        //info!("After conv1 shape: {:?}", out.shape());
        let out = self.bn1.forward(out);
        //info!("After bn1 shape: {:?}", out.shape());
        let out = self.relu.forward(out);
        //info!("After relu shape: {:?}", out.shape());
        let out = self.tanh.forward(out);
        //info!("After tanh shape: {:?}", out.shape());
        let out = out.flatten(0, 3);
        //info!("After flatten shape: {:?}", out.shape());
        let out = self.linear.forward(out);
        //info!("After linear shape: {:?}", out.shape());
        let out = sigmoid(out);
        //info!("Value head output shape: {:?}", out.shape());
        out
    }
}

#[derive(Module, Debug)]
pub struct PolicyHead<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B>,
    relu: Relu,
    linear: Linear<B>,
}

impl<B: Backend> PolicyHead<B> {
    pub fn new(channels: usize, device: &Device<B>) -> Self {
        // conv1x1
        let conv1 = Conv2dConfig::new([channels, channels], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Same)
            .with_bias(false)
            .init(device);
        let bn1 = BatchNormConfig::new(channels).init(device);
        let relu = Relu::new();
        let linear = LinearConfig::new(channels * 3 * 3, 200).init(device);
        Self {
            conv1,
            bn1,
            relu,
            linear,
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 1> {
        //info!("Policy head forward pass with input shape: {:?}", input.shape());
        let out = self.conv1.forward(input);
        //info!("After conv1 shape: {:?}", out.shape());
        let out = self.bn1.forward(out);
        //info!("After bn1 shape: {:?}", out.shape());
        let out = self.relu.forward(out);
        //info!("After relu shape: {:?}", out.shape());
        let out = out.flatten(0, 3);
        //info!("After flatten shape: {:?}", out.shape());
        let out = self.linear.forward(out);
        //info!("After linear shape: {:?}", out.shape());
        let out = softmax(out, 0);
        //info!("Policy head output shape: {:?}", out.shape());
        out
    }
}
