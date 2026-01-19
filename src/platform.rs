use wasm_bindgen::prelude::*;
use log::info;

pub trait Platform: Clone {
    fn now() -> f64;
    fn random() -> f32;
    fn print(str_print: &str);
}

#[derive(Clone)]
pub struct WasmPlatform;

impl Platform for WasmPlatform {
    fn now() -> f64 {
        #[wasm_bindgen(js_namespace = performance)]
        extern "C" {
            fn now() -> f64;
        }
        now()
    }

    fn random() -> f32 {
        #[wasm_bindgen(js_namespace = Math)]
        extern "C" {
            fn random() -> f64;
        }
        random() as f32
    }

    fn print(str_print: &str) {
        info!("{}", str_print);
    }
}

#[derive(Clone)]
pub struct NativePlatform;

impl Platform for NativePlatform {
    fn now() -> f64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64() * 1000.0
    }

    fn random() -> f32 {
        use rand::Rng;
        rand::rng().random()
    }

    fn print(str_print: &str) {
        println!("{}", str_print);
    }
}
