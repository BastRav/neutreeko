use super::{ANN, ANNConfig, PolicyValueOutput};
use burn::{
    nn::loss::{MseLoss, Reduction},
    optim::{Adam, AdamConfig, GradientsParams, Optimizer, adaptor::OptimizerAdaptor},
    tensor::{Device, Tensor, activation::log_softmax, backend::AutodiffBackend},
};

pub struct PolicyValueTarget<B: AutodiffBackend> {
    pub value: Tensor<B, 1>,
    pub policy: Tensor<B, 2>,
}

pub struct ANNTrainer<B: AutodiffBackend> {
    pub model: ANN<B>,
    pub optimizer: OptimizerAdaptor<Adam, ANN<B>, B>,
    pub device: Device<B>,
}

impl<B: AutodiffBackend<FloatElem = f32>> ANNTrainer<B> {
    pub fn new() -> Self {
        let device = B::Device::default();
        let model = ANNConfig::init(32, &device);
        let optimizer = AdamConfig::new().init();

        Self {
            model,
            optimizer,
            device,
        }
    }

    pub fn loss(&self, output: PolicyValueOutput<B>, target: PolicyValueTarget<B>, illegal_mask: Tensor<B, 2>) -> Tensor<B, 1> {
        let masked_probabilities = output.policy + illegal_mask;
        let log_probabilities = log_softmax(masked_probabilities, 1);
        let policy_loss = -(target.policy * log_probabilities).sum_dim(1).mean();
        let value_loss = MseLoss::new().forward(output.value, target.value, Reduction::Mean);
        policy_loss + value_loss * 0.5
    }

    pub fn train_step(&mut self, input:Tensor<B, 3>, target: PolicyValueTarget<B>, illegal_mask: Tensor<B, 2>) -> Tensor<B, 1> {
        // Forward pass
        let output = self.model.forward(input);

        let loss = self.loss(output, target, illegal_mask);
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.model);

        // Update model parameters
        self.model = self.optimizer.step(0.001, self.model.clone(), grads);
        loss
    }

    pub fn save(&self, _path: &str) {
        // Implement model saving
    }

    pub fn load(&mut self, _path: &str) {
        // Implement model loading
    }
}
