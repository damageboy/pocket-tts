extern crate self as pocket_tts_cli;

pub mod commands;
pub mod server;
pub mod voice;

#[cfg(test)]
#[path = "../tests/base64_tests.rs"]
mod base64_tests;

#[cfg(test)]
#[path = "../tests/server_tests.rs"]
mod server_tests;

#[cfg(test)]
#[path = "../tests/stream_tests.rs"]
mod stream_tests;
